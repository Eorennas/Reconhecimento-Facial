import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw , ImageFont
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

def treinamento(dados_path, modelo_salvo_path=None, n_vizinhos=None, knn_algo='ball_tree', verbose=False):

    X = []
    y = []

    for sub_pasta_path in os.listdir(dados_path):
        if not os.path.isdir(os.path.join(dados_path, sub_pasta_path)):
            continue

        for image_path in image_files_in_folder(os.path.join(dados_path, sub_pasta_path)):
            image = face_recognition.load_image_file(image_path)
            face_localizadas = face_recognition.face_locations(image)

            if len(face_localizadas) == 1:
                X.append(face_recognition.face_encodings(image, known_face_locations=face_localizadas, model="cnn")[0])
                y.append(sub_pasta_path)

    if n_vizinhos is None:
        n_vizinhos = int(round(math.sqrt(len(X))))

    modelo_carregado = neighbors.KNeighborsClassifier(n_neighbors=n_vizinhos, algorithm=knn_algo, weights='distance')
    modelo_carregado.fit(X, y)

    if modelo_salvo_path is not None:
        with open(modelo_salvo_path, 'wb') as f:
            pickle.dump(modelo_carregado, f)

    return modelo_carregado


def predict(X_img_path, modelo_path=None, limiar_de_distancia=0.6):
    with open(modelo_path, 'rb') as f:
        modelo_carregado = pickle.load(f)

    img = face_recognition.load_image_file(X_img_path)
    face_localizadas = face_recognition.face_locations(img)

    if len(face_localizadas) == 0:
        return []

    face_codificadas = face_recognition.face_encodings(img, known_face_locations=face_localizadas, model="cnn")

    distancia_mais_proxima = modelo_carregado.kneighbors(face_codificadas, n_neighbors=1)
    sao_compativel = [distancia_mais_proxima[0][i][0] <= limiar_de_distancia for i in range(len(face_localizadas))]

    return [(pred, loc) if rec else ("Desconhecido", loc) for pred, loc, rec in zip(modelo_carregado.predict(face_codificadas), face_localizadas, sao_compativel)]

def mostrar_resultados_de_prediccao(image_path, prediccao):

    pil_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    fonte = ImageFont.truetype("C:\\Users\\rennan\\Documents\\GitHub\\Reconhecimento-Facial\\facial\\Arial.ttf", 25)

    for nome, (topo, direita, baixo, left) in prediccao:

        draw.rectangle(((left, topo), (direita, baixo)), outline=(0, 0, 255))
        draw.rectangle(((left, baixo - 37 - 10), (direita, baixo)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, baixo - 37 - 5), nome, fill=(255, 255, 255, 255), font=fonte)

    del draw

    pil_image.show()


if __name__ == "__main__":

    print("treinando classificador KNN...")
    classificador = treinamento("c:/Users/rennan/Documents/GitHub/Reconhecimento-Facial/facial/exemplos/dados", modelo_salvo_path="modeloTreinado.clf", n_vizinhos=2)
    print("Treino completo")

    for image in os.listdir("c:/Users/rennan/Documents/GitHub/Reconhecimento-Facial/facial/exemplos/teste"):
        full_file_path = os.path.join("c:/Users/rennan/Documents/GitHub/Reconhecimento-Facial/facial/exemplos/teste", image)
        prediccao = predict(full_file_path, modelo_path="C:\\Users\\rennan\\Documents\\GitHub\\Reconhecimento-Facial\\facial\\modeloTreinado.clf")
        mostrar_resultados_de_prediccao(os.path.join("c:/Users/rennan/Documents/GitHub/Reconhecimento-Facial/facial/exemplos/teste", image), prediccao)
