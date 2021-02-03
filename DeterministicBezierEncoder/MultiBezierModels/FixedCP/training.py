import torch
import torch.nn.functional as F
import torchvision

from torch.utils.tensorboard import SummaryWriter
from DeterministicBezierEncoder.MultiBezierModels.FixedCP.dataset_generation import bezier
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Utils.chamfer_distance import chamfer_distance
from Utils.probabilistic_map import ProbabilisticMap
from itertools import permutations


def intersection_over_union(predicted, target):
    return torch.sum(predicted * target) / torch.sum((predicted + target) - predicted * target)

def train_one_bezier_transformer(model, dataset, batch_size, num_epochs, optimizer,
                                 num_experiment, lr=1e-4, cuda=True, debug=True):
    print("\n\nTHE TRAINING BEGINS")
    print("Experiment #{} ---> batch_size={} num_epochs={} learning_rate={}".format(
        num_experiment,  batch_size, num_epochs, lr))

    # basedir = "/data1slow/users/asuso/trans_bezier"
    basedir = "/home/albert/PycharmProjects/trans_bezier"

    probabilistic_map_generator = ProbabilisticMap((model.image_size, model.image_size, 50))
    cp_covariance = torch.tensor([[[3, 0], [0, 3]] for i in range(model.num_cp)], dtype=torch.float32)
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
        writer = SummaryWriter(basedir+"/graphics/DeterministicBezierEncoder/MultiBezierModels/FixedCP/"+str(model.num_cp)+"CP_maxBeziers"+str(model.max_beziers)+"exp"+str(num_experiment))
        counter = 0

    # Separamos el dataset en imagenes y secuencias
    images, sequences, tgt_padding_masks = dataset
    # Enviamos los datos y el modelo a la GPU
    if cuda:
        images = images.cuda()
        sequences = sequences.cuda()
        tgt_padding_masks = tgt_padding_masks.cuda()
        model = model.cuda()

    # Particionamos el dataset en training y validation
    # images.shape=(N, 1, 64, 64)
    # sequences.shape=(100, N, 2)
    im_training = images[:40000]
    im_validation = images[40000:]
    seq_training = sequences[:, :40000]
    seq_validation = sequences[:, 40000:]
    tgt_padding_masks_training = tgt_padding_masks[:40000]
    tgt_padding_masks_validation = tgt_padding_masks[40000:]

    # Definimos el optimizer
    optimizer = optimizer(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=10**(-0.5), patience=8, min_lr=1e-8)

    for epoch in range(num_epochs):
        print("Beginning epoch number", epoch+1)
        for i in range(0, len(im_training)-batch_size+1, batch_size):
            # Obtenemos el batch
            im = im_training[i:i+batch_size]
            seq = seq_training[:, i:i+batch_size]
            tgt_padding_masks = tgt_padding_masks_training[i:i+batch_size]

            # Ejecutamos el modelo sobre el batch
            probabilities = model(im, seq, tgt_padding_masks)

            #probabilities.shape = (tgt_seq_len, batch_size, num_probabilites)
            #seq.shape = (tgt_seq_len, batch_size, 1)
            # Calculamos la loss
            loss = 0
            for k in range(batch_size):
                # Obtenemos el numero de tokens que forman la secuencia
                num_tokens = tgt_padding_masks.shape[1] - torch.sum(tgt_padding_masks[k])

                # Bajo la suposicion de que cada curva de bezier esta formada por num_cp+1 tokens, calculamos
                # todas las permutaciones posibles de las curvas
                perm = permutations(range(num_tokens//(model.num_cp+1)))
                perm = torch.tensor([p for p in perm], dtype=torch.long)

                # Creamos una "permutacion de tokens" a partir de la "permutacion de curvas de bezier"
                perm_tokens = torch.empty((perm.shape[0], num_tokens), dtype=torch.long)
                for j in range(perm.shape[1]):
                    for s in range(model.num_cp+1):
                        perm_tokens[:, j*(model.num_cp+1) + s] = (model.num_cp+1)*perm[:, j] + s
                # Calculamos la permutacion que invierte el orden de los CP de cada curva
                inverted_perm_tokens = perm_tokens.clone().detach()
                for n in range(perm.shape[1]):
                    inverted_perm_tokens[:, n*(3+1):n*(3+1)+3] = perm_tokens[:, n*(3+1):n*(3+1)+3].flip(1)

                lower_loss = float('inf')
                # Calculamos las loss de cada permutacion y nos quedamos con la mas baja
                for permutation, inverted_permutation in zip(perm_tokens, inverted_perm_tokens):
                    # Calculamos la loss asociada a los CP de cada curva sin invertir
                    actual_loss = F.cross_entropy(probabilities[permutation, k], seq[:num_tokens, k])
                    if actual_loss < lower_loss:
                        lower_loss = actual_loss
                    # Calculamos la loss asociada a los CP de cada curva invertidos
                    actual_loss = F.cross_entropy(probabilities[inverted_permutation, k], seq[:num_tokens, k])
                    if actual_loss < lower_loss:
                        lower_loss = actual_loss
                loss += lower_loss

            loss = loss/batch_size

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
                seq = seq_validation[:, j:j + batch_size]
                tgt_padding_masks = tgt_padding_masks_validation[j:j + batch_size]

                # Ejecutamos el modelo sobre el batch
                probabilities = model(im, seq, tgt_padding_masks)

                # probabilities.shape = (tgt_seq_len, batch_size, num_probabilites)
                # seq.shape = (tgt_seq_len, batch_size, 1)
                # Calculamos la loss
                loss = 0
                for k in range(batch_size):
                    # Obtenemos el numero de tokens que forman la secuencia
                    num_tokens = tgt_padding_masks.shape[1] - torch.sum(tgt_padding_masks[k])

                    # Bajo la suposicion de que cada curva de bezier esta formada por num_cp+1 tokens, calculamos
                    # todas las permutaciones posibles de las curvas
                    perm = permutations(range(num_tokens // (model.num_cp + 1)))
                    perm = torch.tensor([p for p in perm], dtype=torch.long)

                    # Creamos una "permutacion de tokens" a partir de la "permutacion de curvas de bezier"
                    perm_tokens = torch.empty((perm.shape[0], num_tokens), dtype=torch.long)
                    for n in range(perm.shape[1]):
                        for s in range(model.num_cp + 1):
                            perm_tokens[:, n * (model.num_cp + 1) + s] = (model.num_cp + 1) * perm[:, n] + s
                    # Calculamos la permutacion que invierte el orden de los CP de cada curva
                    inverted_perm_tokens = perm_tokens.clone().detach()
                    for n in range(perm.shape[1]):
                        inverted_perm_tokens[:, n * (3 + 1):n * (3 + 1) + 3] = perm_tokens[:,
                                                                               n * (3 + 1):n * (3 + 1) + 3].flip(1)

                    lower_loss = float('inf')
                    # Calculamos las loss de cada permutacion y nos quedamos con la mas baja
                    for permutation, inverted_permutation in zip(perm_tokens, inverted_perm_tokens):
                        # Calculamos la loss asociada a los CP de cada curva sin invertir
                        actual_loss = F.cross_entropy(probabilities[permutation, k], seq[:num_tokens, k])
                        if actual_loss < lower_loss:
                            lower_loss = actual_loss
                        # Calculamos la loss asociada a los CP de cada curva invertidos
                        actual_loss = F.cross_entropy(probabilities[inverted_permutation, k], seq[:num_tokens, k])
                        if actual_loss < lower_loss:
                            lower_loss = actual_loss
                    loss += lower_loss

                loss = loss / batch_size

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
                torch.save(model.state_dict(), basedir+"/state_dicts/DeterministicBezierEncoder/MultiBezierModels/FixedCP/"+str(model.num_cp)+"CP_maxBeziers"+str(model.max_beziers)+"exp"+str(num_experiment))
            cummulative_loss = 0

            # Iniciamos la evaluación del modo "predicción"
            iou_value = 0
            chamfer_value = 0
            probabilistic_similarity = 0
            # Inicialmente, predeciremos 10 imagenes que almacenaremos en tensorboard
            target_images = torch.empty((10, 1, 64, 64))
            predicted_images = torch.empty((10, 1, 64, 64))
            for idx in range(0, 200, 20):
                tgt_im = im_validation[idx].unsqueeze(0)
                pred_im = torch.zeros_like(tgt_im)

                control_points = model.predict(tgt_im)

                resolution = 150
                # Separamos los CP en curvas de bezier y las vamos pintando
                bezier_start = 0
                for i, cp in enumerate(control_points):
                    if cp == tgt_im.shape[1]*tgt_im.shape[1]:
                        output = bezier(control_points[bezier_start:i].unsqueeze(1),
                                        torch.tensor([i-bezier_start], dtype=torch.long, device=control_points.device),
                                        torch.linspace(0, 1, resolution, device=control_points.device).unsqueeze(0),
                                        device=control_points.device)
                        pred_im[0, 0, output[0, :, 0], output[0, :, 1]] = 1

                        bezier_start = i+1

                iou_value += intersection_over_union(pred_im, tgt_im)
                chamfer_value += chamfer_distance(pred_im[0].cpu().numpy(), tgt_im[0].cpu().numpy())

                if control_points.shape[0] > 0:
                    probability_map = probabilistic_map_generator(control_points.unsqueeze(1),
                                                                  len(control_points)*torch.ones(1, dtype=torch.long, device=control_points.device),
                                                                  cp_covariances)
                    reduced_map, _ = torch.max(probability_map, dim=3)
                    probabilistic_similarity += torch.sum(reduced_map*tgt_im)/torch.sum(tgt_im)

                target_images[idx//20] = tgt_im.unsqueeze(0)
                predicted_images[idx // 20] = pred_im.unsqueeze(0)

            #Guardamos estas primeras 10 imagenes en tensorboard
            img_grid = torchvision.utils.make_grid(target_images)
            writer.add_image('target_images', img_grid)
            img_grid = torchvision.utils.make_grid(predicted_images)
            writer.add_image('predicted_images', img_grid)

            # Predecimos 490 imagenes mas, esta vez almacenando solo el error
            for idx in range(200, 10000, 20):
                tgt_im = im_validation[idx].unsqueeze(0)
                pred_im = torch.zeros_like(tgt_im)

                control_points = model.predict(tgt_im)

                resolution = 150
                # Separamos los CP en curvas de bezier y las vamos pintando
                bezier_start = 0
                for i, cp in enumerate(control_points):
                    if cp == tgt_im.shape[1] * tgt_im.shape[1]:
                        output = bezier(control_points[bezier_start:i].unsqueeze(1),
                                        torch.tensor([i - bezier_start], dtype=torch.long,
                                                     device=control_points.device),
                                        torch.linspace(0, 1, resolution, device=control_points.device).unsqueeze(0),
                                        device=control_points.device)
                        pred_im[0, 0, output[0, :, 0], output[0, :, 1]] = 1

                        bezier_start = i + 1

                iou_value += intersection_over_union(pred_im, tgt_im)
                chamfer_value += chamfer_distance(pred_im[0].cpu().numpy(), tgt_im[0].cpu().numpy())

                if control_points.shape[0] > 0:
                    probability_map = probabilistic_map_generator(control_points.unsqueeze(1),
                                                                  len(control_points)*torch.ones(1, dtype=torch.long, device=control_points.device),
                                                                  cp_covariances)
                    reduced_map, _ = torch.max(probability_map, dim=3)
                    probabilistic_similarity += torch.sum(reduced_map * tgt_im)/torch.sum(tgt_im)

            # Guardamos el error de predicción en tensorboard
            writer.add_scalar("Prediction/IoU", iou_value/500, counter)
            writer.add_scalar("Prediction/Chamfer_distance", chamfer_value / 500, counter)
            writer.add_scalar("Prediction/Probabilistic_similarity", probabilistic_similarity / 500, counter)

        # Volvemos al modo train para la siguiente epoca
        model.train()