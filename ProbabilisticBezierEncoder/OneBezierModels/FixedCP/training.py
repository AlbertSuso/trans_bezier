import torch
import torchvision
import numpy as np
import time

from torch.utils.tensorboard import SummaryWriter
from ProbabilisticBezierEncoder.OneBezierModels.FixedCP.dataset_generation import bezier
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Utils.chamfer_distance import chamfer_distance, generate_loss_images
from Utils.probabilistic_map import ProbabilisticMap
from ProbabilisticBezierEncoder.OneBezierModels.FixedCP.losses import dmap_loss, pmap_loss


def intersection_over_union(predicted, target):
    return torch.sum(predicted * target) / torch.sum((predicted + target) - predicted * target)

def step_decay(original_cp_variance, epoch, var_drop=0.5, epochs_drop=5, min_var=0.1):
    if epoch < 2*epochs_drop:
        return torch.tensor([original_cp_variance])
    return max(torch.tensor([min_var]), original_cp_variance * (var_drop ** torch.floor(torch.tensor([(epoch-epochs_drop) / epochs_drop]))))

def train_one_bezier_transformer(model, dataset, batch_size, num_epochs, optimizer, loss_mode,
                                 num_experiment, cp_variance, var_drop, epochs_drop, min_variance, penalization_coef,
                                 lr=1e-4, cuda=True, debug=True):
    # torch.autograd.set_detect_anomaly(True)
    print("\n\nTHE TRAINING BEGINS")
    print("Experiment #{} ---> loss={} distance_loss={} batch_size={} num_epochs={} learning_rate={} cp_variance={} var_drop={} epochs_drop={} min_var={} pen_coef={}".format(
        loss_mode[0], loss_mode[1], num_experiment,  batch_size, num_epochs, lr, cp_variance, var_drop, epochs_drop, min_variance, penalization_coef))

    # basedir = "/data1slow/users/asuso/trans_bezier"
    basedir = "/home/asuso/PycharmProjects/trans_bezier"

    # Iniciamos una variable en la que guardaremos la mejor loss obtenida en validation
    best_loss = float('inf')

    # Inicializamos el writer de tensorboard, y las variables que usaremos para
    # la recopilación de datos.
    cummulative_loss = 0
    if debug:
        # Tensorboard writter
        writer = SummaryWriter(basedir + "/graphics/ProbabilisticBezierEncoder/OneBezierModels/FixedCP/loss"+str(loss_mode[0]+"_distance"+str(loss_mode[1])))
        counter = 0

    # Obtenemos las imagenes del dataset
    images = dataset

    assert loss_mode[0] in ['pmap', 'dmap']
    assert loss_mode[1] in ['l2', 'quadratic', 'exp']
    # En caso de que la loss seleccionada sea la del mapa probabilístico, hacemos los preparativos necesarios
    if loss_mode[0] == 'pmap':
        # Inicializamos el generador de mapas probabilisticos y la matriz de covariancias
        probabilistic_map_generator = ProbabilisticMap((model.image_size, model.image_size, 50))
        cp_covariance = torch.tensor([ [[1, 0], [0, 1]] for i in range(model.num_cp)], dtype=torch.float32)
        cp_covariances = torch.empty((model.num_cp, batch_size, 2, 2))
        for i in range(batch_size):
            cp_covariances[:, i, :, :] = cp_covariance

        # Obtenemos las imagenes del groundtruth con el factor de penalización añadido
        loss_images = generate_loss_images(images, weight=penalization_coef, distance=loss_mode[1])
        if cuda:
            probabilistic_map_generator = probabilistic_map_generator.cuda()
            cp_covariances = cp_covariances.cuda()
            images = images.cuda()
            loss_images = loss_images.cuda()
            model = model.cuda()

    # En caso de que la loss seleccionada sea la del mapa de distancias, hacemos los preparativos necesarios
    if loss_mode[0] == 'dmap':
        grid = torch.empty((1, 1, images.shape[2], images.shape[3], 2), dtype=torch.float32)
        for i in range(images.shape[2]):
            grid[0, 0, i, :, 0] = i
            grid[0, 0, :, i, 1] = i
        if cuda:
            images = images.cuda()
            grid = grid.cuda()
            model = model.cuda()

    # Particionamos el dataset en training y validation
    # images.shape=(N, 1, 64, 64)
    im_training = images[:40000]
    im_validation = images[40000:]
    if loss_mode[0] == 'pmap':
        loss_im_training = loss_images[:40000]
        loss_im_validation = loss_images[40000:]

    # Definimos el optimizer
    optimizer = optimizer(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=10**(-0.5), patience=8, min_lr=1e-8)

    for epoch in range(num_epochs):
        t0 = time.time()
        print("Beginning epoch number", epoch+1)
        if loss_mode[0] == 'pmap':
            actual_covariances = cp_covariances * step_decay(cp_variance, epoch, var_drop, epochs_drop, min_variance).to(cp_covariances.device)
        for i in range(0, len(im_training)-batch_size+1, batch_size):
            # Obtenemos el batch
            im = im_training[i:i+batch_size]

            # Ejecutamos el modelo sobre el batch
            control_points, num_cps = model(im)
            # control_points, num_cps, variances = model(im)
            # actual_covariances = variances * cp_covariances

            # Calculamos la loss
            if loss_mode[0] == 'pmap':
                loss_im = loss_im_training[i:i + batch_size]
                loss = pmap_loss(control_points, num_cps, actual_covariances, im, loss_im, probabilistic_map_generator)
            elif loss_mode[0] == 'dmap':
                loss = dmap_loss(control_points, num_cps, im, grid, distance=loss_mode[1])
            if debug:
                cummulative_loss += loss

            # Realizamos backpropagation y un paso de descenso del gradiente
            loss.backward()
            optimizer.step()
            model.zero_grad()

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
        model.eval()
        with torch.no_grad():
            cummulative_loss = 0
            for j in range(0, len(im_validation) - batch_size + 1, batch_size):
                # Obtenemos el batch
                im = im_validation[j:j+batch_size]

                # Ejecutamos el modelo sobre el batch
                control_points, num_cps = model(im)

                # Calculamos la loss
                if loss_mode[0] == 'pmap':
                    loss_im = loss_im_validation[j:j + batch_size]
                    loss = pmap_loss(control_points, num_cps, actual_covariances, im, loss_im,
                                     probabilistic_map_generator)
                elif loss_mode[0] == 'dmap':
                    loss = dmap_loss(control_points, num_cps, im, grid, distance=loss_mode[1])
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
                torch.save(model.state_dict(), basedir+"/state_dicts/ProbabilisticBezierEncoder/OneBezierModels/FixedCP/loss"+str(loss_mode[0]+"_distance"+str(loss_mode[1])))
            cummulative_loss = 0

            
            # Iniciamos la evaluación del modo "predicción"
            iou_value = 0
            chamfer_value = 0

            # Inicialmente, predeciremos 10 imagenes que almacenaremos en tensorboard
            target_images = im_validation[0:200:20].cuda()
            predicted_images = torch.zeros_like(target_images)
            control_points, num_cps = model(target_images)
            # Renderizamos las imagenes predichas
            im_seq = bezier(control_points, num_cps, torch.linspace(0, 1, 150, device=control_points.device).unsqueeze(0), device='cuda')
            im_seq = torch.round(im_seq).long()
            for i in range(10):
                predicted_images[i, 0, im_seq[i, :, 0], im_seq[i, :, 1]] = 1

            # Guardamos estas primeras 10 imagenes en tensorboard
            img_grid = torchvision.utils.make_grid(target_images)
            writer.add_image('target_images', img_grid)
            img_grid = torchvision.utils.make_grid(predicted_images)
            writer.add_image('predicted_images', img_grid)

            # Calculamos metricas
            iou_value += intersection_over_union(predicted_images, target_images)
            chamfer_value += np.sum(chamfer_distance(predicted_images[:, 0].cpu().numpy(), target_images[:, 0].cpu().numpy()))


            # Finalmente, predecimos 490 imagenes mas para calcular IoU y chamfer_distance
            target_images = im_validation[200:10000:20].cuda()
            predicted_images = torch.zeros_like(target_images)
            control_points, num_cps = model(target_images)
            # Renderizamos las imagenes predichas
            im_seq = bezier(control_points, num_cps, torch.linspace(0, 1, 150, device=control_points.device).unsqueeze(0), device='cuda')
            im_seq = torch.round(im_seq).long()
            for i in range(490):
                predicted_images[i, 0, im_seq[i, :, 0], im_seq[i, :, 1]] = 1
            # Calculamos metricas
            iou_value += intersection_over_union(predicted_images, target_images)
            chamfer_value += np.sum(chamfer_distance(predicted_images[:, 0].cpu().numpy(), target_images[:, 0].cpu().numpy()))

            # Guardamos los resultados en tensorboard
            writer.add_scalar("Prediction/IoU", iou_value / 500, counter)
            writer.add_scalar("Prediction/Chamfer_distance", chamfer_value / 500, counter)

        # Volvemos al modo train para la siguiente epoca
        model.train()
        print("Tiempo por epoca de", time.time()-t0)
