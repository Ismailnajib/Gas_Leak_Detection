# 🔥 Gas Leak Detection System

## 🧾 Introduction :

Dans les environnements industriels, la présence de gaz toxiques constitue une menace sérieuse pour la santé humaine ainsi que pour la sécurité des opérations. Les fuites de gaz non détectées peuvent entraîner des conséquences graves, telles que des intoxications, des explosions ou encore des impacts environnementaux importants.

Les méthodes traditionnelles de détection reposent principalement sur l’utilisation de capteurs chimiques. Bien qu'efficaces, ces capteurs présentent certaines limites, notamment en ce qui concerne la surveillance en temps réel, les coûts de maintenance élevés et la capacité à couvrir efficacement l’ensemble des zones à risque.


---

## 📌 Fonctionnalités

- Détection de fuite de gaz en temps réel
- Alerte visuelle
- Utilisation d’un dashboard de surveillance
- Intégration possible de plusieurs caméras IP
- Journalisation et enregistrement des événements de détection

---

## 🛠️ Technologies utilisées

- Python 3.x
- Colab
- HTML
- CSS
- OpenCV      
- JavaScript
- fastapi
- utralytics
- websockets

---
## 🏗️ Architecture du Projet :
Ce projet permet de détecter les fuites de gaz en utilisant des algorithmes de vision par ordinateur, intégrés dans un système complet avec un tableau de bord (dashboard) dédié. Celui-ci permet de superviser et de contrôler plusieurs caméras thermiques déployées dans un environnement industriel. Le système reçoit en temps réel les flux vidéo provenant de toutes les caméras installées sur le site. Ces flux sont ensuite traités par le modèle de vision par ordinateur, qui effectue l'inférence pour détecter d'éventuelles fuites de gaz. Lorsqu'une fuite est identifiée, le modèle renvoie les coordonnées des boîtes englobantes ainsi que les scores de confiance vers la caméra concernée. Par ailleurs, une version de la vidéo détectée est automatiquement sauvegardée localement sur le PC, assurant ainsi une traçabilité des incidents.

![Screenshot (1)](https://github.com/Ismailnajib/Gas_Leak_Detection/blob/main/Project_Arch%20(1).jpg)  


---
##  Modèle YOLOv11n :

Pour l'entraînement du modèle de détection des fuites de gaz, nous avons utilisé le modèle pré-entraîné YOLOv11n, reconnu pour son efficacité en termes de précision et de performances en temps réel. Dans un premier temps, l'entraînement a été effectué à l'aide d'un jeu de données simulé représentant des fumées, permettant au modèle d'apprendre les premières caractéristiques visuelles associées aux fuites. Ensuite, une phase de finetuning a été réalisée en utilisant un jeu de données réel composé d'images capturées par des caméras thermiques de type OGI (Optical Gas Imaging). Cette approche en deux étapes permet d'améliorer la robustesse du modèle en combinant des données synthétiques et réelles, tout en optimisant ses performances pour la détection en conditions réelles.

![Screenshot (1)](https://github.com/EmbeddiaInnovX/ComputerVision_Based_AQS/blob/main/Train_Arch.jpg)  

Après l'entraînement initial du modèle avec un jeu de données simulé composé de plus de 50k images représentant des fumées artificielles, nous avons procédé à une phase de finetuning à l'aide de données réelles. Cette deuxième base de données contient plus de 7k images capturées à l’aide de caméras thermiques OGI (Optical Gas Imaging), dans des conditions industrielles réelles. Chaque image a été annotée manuellement afin d'assurer une qualité d’apprentissage optimale. Le finetuning du modèle a été réalisé sur 350 époques, ce qui a permis d’adapter efficacement le modèle pré-entraîné aux particularités visuelles des fuites de gaz réelles, tout en améliorant sa robustesse et sa capacité de généralisation.


![Screenshot (1)](https://github.com/EmbeddiaInnovX/ComputerVision_Based_AQS/blob/main/YOLOv11n_Train_Plots.png)  

### Résultats finaux du modèle (YOLOv11n)

| Précision | Rappel | mAP@0.5 | mAP@0.5:0.95 | Latence |
|-----------|--------|---------|--------------|---------|
| 93,9 %    | 93,0 % | 97,6 %  | 72,4 %       | 1,67 ms |

---
## Dépoiement du Modèle :

Pour le déploiement, la plateforme reçoit le flux vidéo en continu, le transmet au modèle YOLO v11n, lequel traite chaque image en temps réel afin de détecter la présence éventuelle d’une fuite de gaz.

![Screenshot (1)](https://github.com/EmbeddiaInnovX/ComputerVision_Based_AQS/blob/main/Flowchart.jpg)  


## ⚙️ Installation

### 1. Cloner le projet

```bash
git clone https://github.com/EmbeddiaInnovX/ComputerVision_Based_AQS.git
cd ComputerVision_Based_AQS
cd Gas_Leak_Detection
pip install -r requirements.txt
uvicorn app:app --reload
Accéder à l'application:  http://127.0.0.1:8000

```
