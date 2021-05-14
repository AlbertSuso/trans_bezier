import torch
import torchvision
import numpy as np
import time

from torch.utils.tensorboard import SummaryWriter
from ProbabilisticBezierEncoder.MultiBezierModels.FixedCP.dataset_generation import bezier
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Utils.chamfer_distance import chamfer_distance, generate_distance_images
from Utils.probabilistic_map import ProbabilisticMap
from ProbabilisticBezierEncoder.MultiBezierModels.ParallelVersion.losses import loss_function, new_loss


def intersection_over_union(predicted, target):
    return torch.sum(predicted * target) / torch.sum((predicted + target) - predicted * target)

def step_decay(original_cp_variance, epoch, var_drop=0.5, epochs_drop=5, min_var=0.1):
    if epoch < 2*epochs_drop:
        return torch.tensor([original_cp_variance])
    return max(torch.tensor([min_var]), original_cp_variance * (var_drop ** torch.floor(torch.tensor([(epoch-epochs_drop) / epochs_drop]))))

def train_one_bezier_transformer(model, dataset, batch_size, num_epochs, optimizer, num_experiment, lr=1e-4,
                                 loss_type='probabilistic', dataset_name="MNIST", cuda=True, debug=True): # rep_coef=0.1, dist_thresh=4.5, second_term=True
    # torch.autograd.set_detect_anomaly(True)
    print("\n\nTHE TRAINING BEGINS")
    print("MultiBezier Experiment #{} ---> num_cp={} max_beziers={} batch_size={} num_epochs={} learning_rate={}".format(
        num_experiment, model.num_cp, model.max_beziers, batch_size, num_epochs, lr))

    # basedir = "/data1slow/users/asuso/trans_bezier"
    basedir = "/home/asuso/PycharmProjects/trans_bezier"

    # Iniciamos una variable en la que guardaremos la mejor loss obtenida en validation
    best_loss = float('inf')

    # Inicializamos el writer de tensorboard, y las variables que usaremos para
    # la recopilación de datos.
    if debug:
        # Tensorboard writter
        writer = SummaryWriter(basedir + "/graphics/ProbabilisticBezierEncoder/MultiBezierModels/ParallelVersion/"+str(dataset_name)+"_"+str(loss_type)+"_"+str(model.num_cp)+"CP_maxBeziers"+str(model.max_beziers))
        counter = 0

    # Obtenemos las imagenes del dataset
    images = dataset

    if loss_type == "probabilistic":
        # Inicializamos el generador de mapas probabilisticos y la matriz de covariancias
        probabilistic_map_generator = ProbabilisticMap((model.image_size, model.image_size, 50))
        cp_covariance = torch.tensor([ [[1, 0], [0, 1]] for i in range(model.num_cp)], dtype=torch.float32)
        covariances = torch.empty((model.num_cp, batch_size, 2, 2))
        for i in range(batch_size):
            covariances[:, i, :, :] = cp_covariance
        # Obtenemos las distance_images
        distance_images = generate_distance_images(images)

        if cuda:
            distance_images = distance_images.cuda()
            probabilistic_map_generator = probabilistic_map_generator.cuda()
            covariances = covariances.cuda()

        distance_im_training = distance_images[:int(0.83*images.shape[0])]
        distance_im_validation = distance_images[int(0.83*images.shape[0]):]

    else:
        # Obtenemos la groundtruth_seq del dataset (bs, max_N, 2)
        im_size = images.shape[-1]
        groundtruth_sequences = torch.empty((images.shape[0], im_size*im_size, 2), dtype=torch.long)
        max_coords = 0
        for i, im in enumerate(images):
            new_coords = torch.nonzero(im[0])
            groundtruth_sequences[i, :new_coords.shape[0]] = new_coords
            if new_coords.shape[0] > max_coords:
                max_coords = new_coords.shape[0]
        groundtruth_sequences = groundtruth_sequences[:, :max_coords]

        if cuda:
            groundtruth_sequences = groundtruth_sequences.cuda()

        groundtruth_seq_training = groundtruth_sequences[:int(0.83*images.shape[0])]
        groundtruth_seq_validation = groundtruth_sequences[int(0.83*images.shape[0]):]

    # Obtenemos el grid
    grid = torch.empty((1, 1, images.shape[2], images.shape[2], 2), dtype=torch.float32)
    for i in range(images.shape[2]):
        grid[0, 0, i, :, 0] = i
        grid[0, 0, :, i, 1] = i

    if cuda:
        images = images.cuda()
        grid = grid.cuda()
        model = model.cuda()

    # Particionamos el dataset en training y validation
    # images.shape=(N, 1, 64, 64)
    im_training = images[:int(0.83*images.shape[0])]
    im_validation = images[int(0.83*images.shape[0]):]

    # Definimos el optimizer
    optimizer = optimizer(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=10**(-0.5), patience=8, min_lr=1e-8)

    for epoch in range(num_epochs):
        t0 = time.time()
        print("Beginning epoch number", epoch+1)
        cummulative_loss = 0
        for i in range(0, len(im_training)-batch_size+1, batch_size):
            # Obtenemos el batch
            im = im_training[i:i+batch_size]#.cuda()

            # Ejecutamos el modelo sobre el batch
            control_points = model(im)

            if loss_type == "probabilistic":
                distance_im = distance_im_training[i:i + batch_size]  # .cuda()
                # Calculamos la loss
                loss = loss_function(control_points, im, distance_im, covariances, probabilistic_map_generator, grid)#repulsion_coef=rep_coef, dist_thresh=dist_thresh, second_term=second_term
            else:
                groundtruth_seq = groundtruth_seq_training[i:i+batch_size]
                loss = new_loss(control_points, im, groundtruth_seq, grid)

            # Realizamos backpropagation y un paso de descenso del gradiente
            loss.backward()
            optimizer.step()
            model.zero_grad()

            if debug:
                cummulative_loss += loss.detach()


        if debug:
            writer.add_scalar("Training/loss", cummulative_loss/(i/batch_size), counter)


        """
        Al completar cada época, probamos el modelo sobre el conjunto de validation. En concreto:
           - Calcularemos la loss del modelo sobre el conjunto de validación
           - Realizaremos 500 predicciones sobre imagenes del conjunto de validación. Generaremos una imagen a partir de la parametrización de la curva de bezier obtenida.
             Calcularemos las metricas IoU, chamfer_distance, y differentiable_chamfer_distance (probabilistic_map)
             asociadas a estas prediciones (comparandolas con el ground truth).
        """
        model.eval()
        with torch.no_grad():
            cummulative_loss = 0
            for j in range(0, len(im_validation)-batch_size+1, batch_size):
                # Obtenemos el batch
                im = im_validation[i:i + batch_size]  # .cuda()

                # Ejecutamos el modelo sobre el batch
                control_points = model(im)

                if loss_type == "probabilistic":
                    distance_im = distance_im_validation[j:j + batch_size]  # .cuda()
                    # Calculamos la loss
                    loss = loss_function(control_points, im, distance_im, covariances, probabilistic_map_generator,
                                         grid)  # repulsion_coef=rep_coef, dist_thresh=dist_thresh, second_term=second_term
                else:
                    groundtruth_seq = groundtruth_seq_validation[j:j + batch_size]
                    loss = new_loss(control_points, im, groundtruth_seq, grid)
                cummulative_loss += loss.detach()

            # Aplicamos el learning rate scheduler
            scheduler.step(cummulative_loss)

            # Recopilamos los datos para tensorboard
            if debug:
                writer.add_scalar("Validation/loss", cummulative_loss/j, counter)

            # Si la loss obtenida es la mas baja hasta ahora, nos guardamos los pesos del modelo
            if cummulative_loss < best_loss:
                print("El modelo ha mejorado!! Nueva loss={}".format(cummulative_loss/j))
                best_loss = cummulative_loss
                torch.save(model.state_dict(), basedir+"/state_dicts/ProbabilisticBezierEncoder/MultiBezierModels/ParallelVersion/"+str(dataset_name)+"_"+str(loss_type)+"_"+str(model.num_cp)+"CP_maxBeziers"+str(model.max_beziers))

            
            # Iniciamos la evaluación del modo "predicción"
            iou_value = 0
            chamfer_value = 0

            # Inicialmente, predeciremos 10 imagenes que almacenaremos en tensorboard
            target_images = im_validation[0:200:20]#.cuda()
            predicted_images = torch.zeros_like(target_images)
            control_points = model(target_images)

            # Renderizamos las imagenes predichas
            for bezier_cp in control_points:
                # Calculamos la secuencia de puntos de esta curva
                num_cps = model.num_cp * torch.ones(10, dtype=torch.long, device=bezier_cp.device)
                im_seq = bezier(bezier_cp, num_cps, torch.linspace(0, 1, 150, device=num_cps.device).unsqueeze(0), device=num_cps.device)
                im_seq = torch.round(im_seq).long()
                for j in range(10):
                    predicted_images[j, 0, im_seq[j, :, 0], im_seq[j, :, 1]] = 1

            # Guardamos estas primeras 10 imagenes en tensorboard
            img_grid = torchvision.utils.make_grid(target_images)
            writer.add_image('target_images', img_grid)
            img_grid = torchvision.utils.make_grid(predicted_images)
            writer.add_image('predicted_images', img_grid)

            # Calculamos metricas
            iou_value += intersection_over_union(predicted_images, target_images)
            chamfer_value += np.sum(chamfer_distance(predicted_images[:, 0].cpu().numpy(), target_images[:, 0].cpu().numpy()))


            # Finalmente, predecimos 490 imagenes mas para calcular IoU y chamfer_distance
            idxs = [200, 1600, 3000, 4400, 5800, 7200, 8600, 10000]
            for i in range(7):
                target_images = im_validation[idxs[i]:idxs[i+1]:20]#.cuda()
                predicted_images = torch.zeros_like(target_images)
                control_points = model(target_images)

                # Renderizamos las imagenes predichas
                for bezier_cp in control_points:
                    # Calculamos la secuencia de puntos de esta curva
                    num_cps = model.num_cp * torch.ones(70, dtype=torch.long, device=bezier_cp.device)
                    im_seq = bezier(bezier_cp, num_cps, torch.linspace(0, 1, 150, device=num_cps.device).unsqueeze(0),
                                    device=num_cps.device)
                    im_seq = torch.round(im_seq).long()
                    for j in range(70):
                        predicted_images[j, 0, im_seq[j, :, 0], im_seq[j, :, 1]] = 1

                # Calculamos metricas
                iou_value += intersection_over_union(predicted_images, target_images)
                chamfer_value += np.sum(chamfer_distance(predicted_images[:, 0].cpu().numpy(), target_images[:, 0].cpu().numpy()))

            # Guardamos los resultados en tensorboard
            writer.add_scalar("Prediction/IoU", iou_value / 500, counter)
            writer.add_scalar("Prediction/Chamfer_distance", chamfer_value / 500, counter)

        # Volvemos al modo train para la siguiente epoca
        model.train()
        print("Tiempo por epoca de", time.time()-t0)
