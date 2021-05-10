import torch
import torchvision
import cv2
import numpy as np
import time

from torch.utils.tensorboard import SummaryWriter
from ProbabilisticBezierEncoder.MultiBezierModels.FixedCP.dataset_generation import bezier
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Utils.chamfer_distance import chamfer_distance, generate_distance_images
from Utils.probabilistic_map import ProbabilisticMap
from ProbabilisticBezierEncoder.MultiBezierModels.SegmentationVersion.losses import loss_function


def intersection_over_union(predicted, target):
    return torch.sum(predicted * target) / torch.sum((predicted + target) - predicted * target)

def step_decay(original_cp_variance, epoch, var_drop=0.5, epochs_drop=5, min_var=0.1):
    if epoch < 2*epochs_drop:
        return torch.tensor([original_cp_variance])
    return max(torch.tensor([min_var]), original_cp_variance * (var_drop ** torch.floor(torch.tensor([(epoch-epochs_drop) / epochs_drop]))))

def train_one_bezier_transformer(model, dataset, batch_size, num_epochs, optimizer, num_experiment, lr=1e-4,
                                cuda=True, debug=True):
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
    cummulative_loss = 0
    if debug:
        # Tensorboard writter
        writer = SummaryWriter(basedir + "/graphics/ProbabilisticBezierEncoder/MultiBezierModels/SegmentationVersion/"+str(model.num_cp)+"CP_maxBeziers"+str(model.max_beziers))
        counter = 0

    # Obtenemos las imagenes del dataset
    images = dataset

    # Realizamos la segmentacion
    # segmented_images = torch.zeros((0, 1, images.shape[-2], images.shape[-1]), dtype=images.dtype, device=images.device)
    connected_components_per_image = 4
    segmented_images = torch.zeros((connected_components_per_image*images.shape[0], 1, images.shape[-2], images.shape[-1]), dtype=images.dtype, device=images.device)
    num_images = 0
    for n, im in enumerate(images):
        num_labels, labels_im = cv2.connectedComponents(im[0].numpy().astype(np.uint8))
        new_segmented_images = torch.zeros((num_labels - 1, 1, images.shape[-2], images.shape[-1]), dtype=images.dtype,
                                           device=images.device)
        for i in range(1, num_labels):
            new_segmented_images[i - 1, 0][labels_im == i] = 1
        #segmented_images = torch.cat((segmented_images, new_segmented_images), dim=0)
        segmented_images[num_images:num_images+num_labels-1] = new_segmented_images
        num_images += num_labels-1
    segmented_images = segmented_images[:num_images]

    # Inicializamos el generador de mapas probabilisticos y la matriz de covariancias
    probabilistic_map_generator = ProbabilisticMap((model.image_size, model.image_size, 50))
    cp_covariance = torch.tensor([ [[1, 0], [0, 1]] for i in range(model.num_cp)], dtype=torch.float32)
    covariances = torch.empty((model.num_cp, batch_size, 2, 2))
    for i in range(batch_size):
        covariances[:, i, :, :] = cp_covariance
    # Obtenemos el grid
    grid = torch.empty((1, 1, images.shape[2], images.shape[2], 2), dtype=torch.float32)
    for i in range(images.shape[2]):
        grid[0, 0, i, :, 0] = i
        grid[0, 0, :, i, 1] = i
    # Obtenemos las distance_images
    distance_images = generate_distance_images(segmented_images)
    if cuda:
        segmented_images = segmented_images.cuda()
        distance_images = distance_images.cuda()
        probabilistic_map_generator = probabilistic_map_generator.cuda()
        covariances = covariances.cuda()
        grid = grid.cuda()
        model = model.cuda()

    # Particionamos el dataset en training y validation
    # images.shape=(N, 1, 64, 64)
    dataset_len = segmented_images.shape[0]
    train_size = int(dataset_len*0.8)
    im_training = segmented_images[:train_size]
    im_validation = segmented_images[train_size:]
    distance_im_training = distance_images[:train_size]
    distance_im_validation = distance_images[train_size:]
    orig_im_validation = images[int(images.shape[0]*0.9):]

    # Definimos el optimizer
    optimizer = optimizer(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=10**(-0.5), patience=8, min_lr=1e-8)

    for epoch in range(num_epochs):
        t0 = time.time()
        print("Beginning epoch number", epoch+1)
        for i in range(0, len(im_training)-batch_size+1, batch_size):
            # Obtenemos el batch
            im = im_training[i:i+batch_size]#.cuda()
            distance_im = distance_im_training[i:i+batch_size]#.cuda()

            # Ejecutamos el modelo sobre el batch
            control_points = model(im)

            # Calculamos la loss
            loss = loss_function(control_points, im, distance_im, covariances, probabilistic_map_generator, grid)
            # Realizamos backpropagation y un paso de descenso del gradiente
            loss.backward()
            optimizer.step()
            model.zero_grad()

            # Recopilación de datos para tensorboard
            k = int(int(im_training.shape[0]/(batch_size*5))*batch_size + 1)
            if debug:
                cummulative_loss += loss.detach()
                if i%k == k-1:
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
            for j in range(0, len(im_validation)-batch_size+1, batch_size):
                # Obtenemos el batch
                im = images[j:j+batch_size]#.cuda()
                distance_im = distance_im_validation[j:j + batch_size]#.cuda()

                # Ejecutamos el modelo sobre el batch
                control_points = model(im)

                # Calculamos la loss
                loss = loss_function(control_points, im, distance_im, covariances, probabilistic_map_generator, grid)
                cummulative_loss += loss.detach()

            # Aplicamos el learning rate scheduler
            scheduler.step(cummulative_loss)

            # Recopilamos los datos para tensorboard
            if debug:
                writer.add_scalar("Validation/loss", cummulative_loss/(j/batch_size+1), counter)

            # Si la loss obtenida es la mas baja hasta ahora, nos guardamos los pesos del modelo
            if cummulative_loss < best_loss:
                print("El modelo ha mejorado!! Nueva loss={}".format(cummulative_loss/(j/batch_size+1)))
                best_loss = cummulative_loss
                torch.save(model.state_dict(), basedir+"/state_dicts/ProbabilisticBezierEncoder/MultiBezierModels/SegmentationVersion/"+str(model.num_cp)+"CP_maxBeziers"+str(model.max_beziers)+"_repCoef")
            cummulative_loss = 0

            
            # Iniciamos la evaluación del modo "predicción"
            iou_value = 0
            chamfer_value = 0

            # Inicialmente, predeciremos 10 imagenes que almacenaremos en tensorboard
            target_images = orig_im_validation[0:100:10]#.cuda()
            predicted_images = model.predict(target_images)

            # Guardamos estas primeras 10 imagenes en tensorboard
            img_grid = torchvision.utils.make_grid(target_images)
            writer.add_image('target_images', img_grid)
            img_grid = torchvision.utils.make_grid(predicted_images)
            writer.add_image('predicted_images', img_grid)

            # Calculamos metricas
            iou_value += intersection_over_union(predicted_images, target_images)
            chamfer_value += np.sum(chamfer_distance(predicted_images[:, 0].cpu().numpy(), target_images[:, 0].cpu().numpy()))


            # Finalmente, predecimos 490 imagenes mas para calcular IoU y chamfer_distance
            idxs = [100, 800, 1500, 2200, 2900, 3600, 4300, 5000]
            for i in range(7):
                target_images = im_validation[idxs[i]:idxs[i+1]:10]#.cuda()
                predicted_images = model.predict(target_images)

                # Calculamos metricas
                iou_value += intersection_over_union(predicted_images, target_images)
                chamfer_value += np.sum(chamfer_distance(predicted_images[:, 0].cpu().numpy(), target_images[:, 0].cpu().numpy()))

            # Guardamos los resultados en tensorboard
            writer.add_scalar("Prediction/IoU", iou_value / 500, counter)
            writer.add_scalar("Prediction/Chamfer_distance", chamfer_value / 500, counter)

        # Volvemos al modo train para la siguiente epoca
        model.train()
        print("Tiempo por epoca de", time.time()-t0)
