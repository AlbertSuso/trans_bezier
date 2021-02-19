import torch
import torchvision
import numpy as np
import time

from torch.utils.tensorboard import SummaryWriter
from ProbabilisticBezierEncoder.OneBezierModels.MultiCP2.dataset_generation import bezier
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Utils.chamfer_distance import chamfer_distance, generate_loss_images
from Utils.probabilistic_map import ProbabilisticMap


def intersection_over_union(predicted, target):
    return torch.sum(predicted * target) / torch.sum((predicted + target) - predicted * target)

def step_decay(original_cp_variance, epoch, var_drop=0.5, epochs_drop=5, min_var=0.1):
    if epoch < 2*epochs_drop:
        return torch.tensor([original_cp_variance])
    return max(torch.tensor([min_var]), original_cp_variance * (var_drop ** torch.floor(torch.tensor([(epoch-epochs_drop) / epochs_drop]))))

def train_one_bezier_transformer(model, dataset, batch_size, num_epochs, optimizer,
                                 num_experiment, cp_variance, var_drop, epochs_drop, min_variance, penalization_coef,
                                 lr=1e-4, cuda=True, debug=True):
    # torch.autograd.set_detect_anomaly(True)
    print("\n\nTHE TRAINING BEGINS")
    print("Experiment #{} ---> batch_size={} num_epochs={} learning_rate={} cp_variance={} var_drop={} epochs_drop={} min_variance={} pen_coef={}".format(
        num_experiment,  batch_size, num_epochs, lr, cp_variance, var_drop, epochs_drop, min_variance, penalization_coef))

    # basedir = "/data1slow/users/asuso/trans_bezier"
    basedir = "/home/asuso/PycharmProjects/trans_bezier"

    # Inicializamos el generador de mapas probabilisticos y la matriz de covariancias
    probabilistic_map_generator = ProbabilisticMap((model.image_size, model.image_size, 50))
    cp_covariance = torch.tensor([ [[1, 0], [0, 1]] for i in range(model.max_cp)], dtype=torch.float32)
    cp_covariances = torch.empty((model.max_cp, batch_size, 2, 2))
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
        writer = SummaryWriter(basedir+"/graphics/ProbabilisticBezierEncoder/OneBezierModels/MultiCP/"+str(model.max_cp)+"CP_decMinvar"+str(min_variance)+"_negativeCoef"+str(penalization_coef)+"secondApproach"+"_solvedMistake2")
        counter = 0

    # Obtenemos las imagenes del dataset
    images = dataset
    # Obtenemos las imagenes para la loss
    loss_images = generate_loss_images(images, weight=penalization_coef)
    # Enviamos los datos y el modelo a la GPU
    if cuda:
        images = images.cuda()
        loss_images = loss_images.cuda()
        model = model.cuda()

    # Particionamos el dataset en training y validation
    # images.shape=(N, 1, 64, 64)
    # sequences.shape=(100, N, 2)
    im_training = images[:40000]
    im_validation = images[40000:]
    loss_im_training = loss_images[:40000]
    loss_im_validation = loss_images[40000:]

    # Definimos el optimizer
    optimizer = optimizer(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=10**(-0.5), patience=8, min_lr=1e-8)

    for epoch in range(num_epochs):
        t0 = time.time()
        print("Beginning epoch number", epoch+1)
        actual_covariances = cp_covariances * step_decay(cp_variance, epoch, var_drop, epochs_drop, min_variance).to(cp_covariances.device)
        for i in range(0, len(im_training)-batch_size+1, batch_size):
            # Obtenemos el batch
            im = im_training[i:i+batch_size]
            loss_im = loss_im_training[i:i + batch_size]

            # Ejecutamos el modelo sobre el batch
            control_points, ncp_probabilities = model(im)
            # assert torch.abs(torch.sum(ncp_probabilities)-ncp_probabilities.shape[1]) < 0.1

            # Calculamos la loss
            loss = 0
            for n, cps in enumerate(control_points):
                # Calculamos el mapa de probabilidades asociado a la curva de bezier probabilistica determinada por los n+2 puntos de control "cps"
                probability_map = probabilistic_map_generator(cps, (n+2)*torch.ones((batch_size,), dtype=torch.long, device=cps.device), actual_covariances)
                reduced_map, _ = torch.max(probability_map, dim=-1)

                #Actualizamos la loss
                loss += ncp_probabilities[n].view(-1, 1, 1)*reduced_map
                # loss += -ncp_probabilities[n]*torch.sum(reduced_map * im[:, 0] / torch.sum(im[:, 0], dim=(1, 2)))
            loss = -torch.sum(loss * loss_im[:, 0] / torch.sum(im[:, 0], dim=(1, 2)).view(-1, 1, 1))

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
            for j in range(0, len(im_validation)-batch_size+1, batch_size):
                im = im_validation[j:j + batch_size]
                loss_im = loss_im_validation[j:j + batch_size]

                # Ejecutamos el modelo sobre el batch
                control_points, ncp_probabilities = model(im)
                # assert torch.abs(torch.sum(ncp_probabilities) - ncp_probabilities.shape[1]) < 0.1

                # Calculamos la loss
                loss = 0
                for n, cps in enumerate(control_points):
                    # Calculamos el mapa de probabilidades asociado a la curva de bezier probabilistica determinada por los n+2 puntos de control "cps"
                    probability_map = probabilistic_map_generator(cps,
                                                                  (n + 2) * torch.ones((batch_size,), dtype=torch.long,
                                                                                       device=cps.device),
                                                                  actual_covariances)
                    reduced_map, _ = torch.max(probability_map, dim=-1)

                    # Actualizamos la loss
                    loss += ncp_probabilities[n].view(-1, 1, 1) * reduced_map
                    # loss += -ncp_probabilities[n]*torch.sum(reduced_map * im[:, 0] / torch.sum(im[:, 0], dim=(1, 2)))
                loss = -torch.sum(loss * loss_im[:, 0] / torch.sum(im[:, 0], dim=(1, 2)).view(-1, 1, 1))

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
                torch.save(model.state_dict(), basedir+"/state_dicts/ProbabilisticBezierEncoder/OneBezierModels/MultiCP/"+str(model.max_cp)+"CP_decMinvar"+str(min_variance)+"_negativeCoef"+str(penalization_coef)+"secondApproach"+"_solvedMistake2")
            cummulative_loss = 0

            
            # Iniciamos la evaluación del modo "predicción"
            iou_value = 0
            chamfer_value = 0
            prob_num_cps = torch.zeros(model.max_cp-1, device=im_validation.device)

            # Inicialmente, predeciremos 10 imagenes que almacenaremos en tensorboard
            target_images = im_validation[0:200:20]
            predicted_images = torch.zeros_like(target_images)

            # Obtenemos los puntos de control con mayor probabilidad
            all_control_points, ncp_probabilities = model(target_images)
            num_cps = torch.argmax(ncp_probabilities, dim=0)

            # Actualizamos la probabilidad de los control points
            for i in range(ncp_probabilities.shape[1]):
                prob_num_cps += ncp_probabilities[:, i]

            control_points = torch.empty_like(all_control_points[0])
            for sample in range(10):
                control_points[:, sample, :] = all_control_points[num_cps[sample], :, sample, :]

            # Renderizamos las imagenes predichas
            im_seq = bezier(control_points, num_cps+2, torch.linspace(0, 1, 150, device=control_points.device).unsqueeze(0), device='cuda')
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
            target_images = im_validation[200:10000:20]
            predicted_images = torch.zeros_like(target_images)

            # Obtenemos los puntos de control con mayor probabilidad
            all_control_points, ncp_probabilities = model(target_images)
            num_cps = torch.argmax(ncp_probabilities, dim=0)

            # Actualizamos la probabilidad de los control points
            for i in range(ncp_probabilities.shape[1]):
                prob_num_cps += ncp_probabilities[:, i]

            control_points = torch.empty_like(all_control_points[0])
            for sample in range(490):
                control_points[:, sample, :] = all_control_points[num_cps[sample], :, sample, :]

            # Renderizamos las imagenes predichas
            im_seq = bezier(control_points, num_cps+2, torch.linspace(0, 1, 150, device=control_points.device).unsqueeze(0), device='cuda')
            for i in range(490):
                predicted_images[i, 0, im_seq[i, :, 0], im_seq[i, :, 1]] = 1
            # Calculamos metricas
            iou_value += intersection_over_union(predicted_images, target_images)
            chamfer_value += np.sum(chamfer_distance(predicted_images[:, 0].cpu().numpy(), target_images[:, 0].cpu().numpy()))

            # Guardamos los resultados en tensorboard
            writer.add_scalar("Prediction/IoU", iou_value / 500, counter)
            writer.add_scalar("Prediction/Chamfer_distance", chamfer_value / 500, counter)
            prob_num_cps = prob_num_cps.cpu()
            probabilities = {str(2+i)+"_cp": prob_num_cps[i]/500 for i in range(model.max_cp-1)}
            writer.add_scalars('num_cp probabilities', probabilities, counter)

        # Volvemos al modo train para la siguiente epoca
        model.train()
        print("Tiempo por epoca de", time.time()-t0)
