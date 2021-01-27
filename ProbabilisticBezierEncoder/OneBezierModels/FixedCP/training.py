import torch
import torchvision
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from ProbabilisticBezierEncoder.OneBezierModels.FixedCP.dataset_generation import bezier
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Utils.chamfer_distance import chamfer_distance
from Utils.probabilistic_map import ProbabilisticMap


def intersection_over_union(predicted, target):
    return torch.sum(predicted * target) / torch.sum((predicted + target) - predicted * target)

def train_one_bezier_transformer(model, dataset, batch_size, num_epochs, optimizer,
                                 num_experiment, lr=1e-4, cuda=True, debug=True):
    # torch.autograd.set_detect_anomaly(True)
    print("\n\nTHE TRAINING BEGINS")
    print("Experiment #{} ---> batch_size={} num_epochs={} learning_rate={}".format(
        num_experiment,  batch_size, num_epochs, lr))

    # basedir = "/data1slow/users/asuso/trans_bezier"
    basedir = "/home/albert/PycharmProjects/trans_bezier"

    # Inicializamos el generador de mapas probabilisticos y la matriz de covariancias
    probabilistic_map_generator = ProbabilisticMap((model.image_size, model.image_size, 50))
    cp_covariance = torch.tensor([[[30, 0], [0, 30]] for i in range(model.num_cp)], dtype=torch.float32)
    cp_covariances = torch.empty((model.num_cp, batch_size, 2, 2))
    for i in range(batch_size):
        cp_covariances[:, i, :, :] = cp_covariance
    if cuda:
        probabilistic_map_generator = probabilistic_map_generator.cuda()
        cp_covariances = cp_covariances.cuda()

    # Iniciamos una variable en la que guardaremos la mejor loss obtenida en validation
    best_loss = float('inf')

    # Inicializamos el writer de tensorboard, y las variables que usaremos para
    # la recopilación de datos.
    cummulative_loss = 0
    if debug:
        # Tensorboard writter
        writer = SummaryWriter(basedir+"/graphics/ProbabilisticBezierEncoder/OneBezierModels/FixedCP/"+str(model.num_cp)+"CP_exp"+str(num_experiment))
        counter = 0

    # Separamos el dataset en imagenes y secuencias
    images, sequences = dataset
    # Enviamos los datos y el modelo a la GPU
    if cuda:
        images = images.cuda()
        model = model.cuda()

    # Particionamos el dataset en training y validation
    # images.shape=(N, 1, 64, 64)
    # sequences.shape=(100, N, 2)
    im_training = images[:40000]
    im_validation = images[40000:]

    # Definimos el optimizer
    optimizer = optimizer(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=10**(-0.5), patience=8, min_lr=1e-8)

    tgt_control_points = 64*torch.rand((3, 64, 2), device='cuda')
    for epoch in range(num_epochs):
        print("Beginning epoch number", epoch+1)
        for i in range(0, len(im_training)-batch_size+1, batch_size):
            # Obtenemos el batch
            im = im_training[i:i+batch_size]

            # Ejecutamos el modelo sobre el batch
            control_points, num_cps = model(im)

            # Calculamos el mapa de probabilidades asociado a la curva de bezier probabilistica
            probability_map = probabilistic_map_generator(control_points, num_cps, cp_covariances)
            reduced_map, _ = torch.max(probability_map, dim=-1)

            #Calculamos la loss
            loss = -torch.sum(reduced_map*im)/torch.sum(im)
            #loss = F.mse_loss(control_points, tgt_control_points)
            print("Iteracion", i / 64)
            print("La loss es", loss)
            assert not torch.isnan(loss), "{}".format(control_points)

            if debug:
                cummulative_loss += loss

            # Realizamos backpropagation y un paso de descenso del gradiente
            loss.backward()
            optimizer.step()
            model.zero_grad()

            num_nan = 0
            for name, param in model.named_parameters():
                num_nan += torch.sum(torch.isnan(param))
                if name == "_decoder._embedder._cp_embedder.weight":
                    print("Los pesos que petan son\n", torch.sum(torch.abs(param)))
                    print("Estos suponen un total de", torch.sum(torch.isnan(param)))
            print("El numero de pesos del modelo que son nan es", num_nan)

            # Recopilación de datos para tensorboard
            k = int(40000/(batch_size*5))
            if debug and i%k == k-1:
                writer.add_scalar("Training/loss", cummulative_loss/k, counter)
                counter += 1
                cummulative_loss = 0


        """
        Al completar cada época, probamos el modelo sobre el conjunto de validation. En concreto:
           - Calcularemos la loss del modelo sobre el conjunto de validación
           - Realizaremos 500 predicciones sobre imagenes del conjunto de validación. Generaremos una imagen a partir de la parametrización de la curva de bezier obtenida.
             Calcularemos las metricas IoU, chamfer_distance, y differentiable_chamfer_distance (probabilistic_map)
             asociadas a estas prediciones (comparandolas con el ground truth).
        """

        """
        print("ENTRAMOS EN EVALUATION")
        model.eval()
        with torch.no_grad():
            cummulative_loss = 0
            for j in range(0, len(im_validation)-batch_size+1, batch_size):
                im = im_training[j:j + batch_size]

                # Ejecutamos el modelo sobre el batch
                control_points, num_cps = model(im)

                probability_map = probabilistic_map_generator(control_points, num_cps, cp_covariances)
                reduced_map, _ = torch.max(probability_map, dim=-1)
                loss = -torch.sum(reduced_map * im)

                cummulative_loss += loss

            # Aplicamos el learning rate scheduler
            scheduler.step(cummulative_loss)

            # Recopilamos los datos para tensorboard
            if debug:
                writer.add_scalar("Validation/loss", cummulative_loss/(j/batch_size+1), counter)

            # Si la loss obtenida es la mas baja hasta ahora, nos guardamos los pesos del modelo
            if cummulative_loss < best_loss:
                print("El modelo ha mejorado!! Nueva loss={}".format(cummulative_loss/(j/batch_size+1)))
                best_loss = cummulative_loss
                torch.save(model.state_dict(), basedir+"/state_dicts/ProbabilisticBezierEncoder/OneBezierModels/FixedCP/"+str(model.num_cp)+"CP_exp"+str(num_experiment))
            else:
                print("El modelo no ha mejorado!! Ultima loss={}".format(cummulative_loss/(j/batch_size+1)))
            cummulative_loss = 0

            
            # Iniciamos la evaluación del modo "predicción"
            iou_value = 0
            chamfer_value = 0

            # Inicialmente, predeciremos 10 imagenes que almacenaremos en tensorboard
            target_images = im_validation[0:200:20]
            predicted_images = torch.zeros_like(target_images)
            control_points, num_cps = model(target_images)
            # Renderizamos las imagenes predichas
            im_seq = bezier(control_points, num_cps, torch.linspace(0, 1, 150, device=control_points.device).unsqueeze(0), device='cuda')
            for i in range(10):
                predicted_images[i, 0, im_seq[i, :, 0], im_seq[i, :, 1]] = 1
            # Calculamos metricas
            iou_value += intersection_over_union(predicted_images, target_images)
            # chamfer_value += chamfer_distance(predicted_images[:, 0].cpu().numpy(), target_images[:, 0].cpu().numpy())
            # Guardamos estas primeras 10 imagenes en tensorboard
            print("Las shapes de las imagenes son")
            print(target_images.shape)
            print(predicted_images.shape)
            img_grid = torchvision.utils.make_grid(target_images)
            writer.add_image('target_images', img_grid)
            img_grid = torchvision.utils.make_grid(predicted_images)
            writer.add_image('predicted_images', img_grid)

            # Finalmente, predecimos 490 imagenes mas para calcular IoU y chamfer_distance
            target_images = im_validation[200:10000:20]
            predicted_images = torch.zeros_like(target_images)
            control_points, num_cps = model(target_images)
            # Renderizamos las imagenes predichas
            im_seq = bezier(control_points, num_cps, torch.linspace(0, 1, 150, device=control_points.device).unsqueeze(0), device='cuda')
            for i in range(490):
                predicted_images[i, 0, im_seq[i, :, 0], im_seq[i, :, 1]] = 1
            # Calculamos metricas
            iou_value += intersection_over_union(predicted_images, target_images)
            # chamfer_value += chamfer_distance(predicted_images[:, 0].cpu().numpy(), target_images[:, 0].cpu().numpy())

            # Guardamos los resultados en tensorboard
            writer.add_scalar("Prediction/IoU", iou_value / 500, counter)
            writer.add_scalar("Prediction/Chamfer_distance", chamfer_value / 500, counter)

        # Volvemos al modo train para la siguiente epoca
        model.train()
        """
        print("SALIMOS DE EVALUATION")
