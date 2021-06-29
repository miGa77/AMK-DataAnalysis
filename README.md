# Die handschriftliche alphanumerische Erkennung
Sequential CNN for Handwritten Alphanumeric Recognition

[![Generic badge](https://img.shields.io/badge/ready-passing-<green>.svg)](https://shields.io/) ![Maintaner](https://img.shields.io/badge/maintainer-kevingostomski-blue)


## Überblick

Die Erkennung handgeschriebener alphanumerischer Ausdrücke ist eine schwierige Aufgabe für die Maschine, da handgeschriebene Ziffern nicht perfekt sind und auf vielen verschiedenen Weisen geschrieben werden können. Die handgeschriebene alphanumerische Ausdruckserkennung ist die Lösung für dieses Problem, die das Bild einer Ziffer verwendet um die im Bild vorhandene Ziffer zu erkennen. Das Projekt ist ein Pythonprogramm, welches die Ausdruckserkennung mithilfe eines *faltendem neuronalem Netzwerk (Englisch =  Convolutional Neural Network -> CNN) löst. Ein CNN ist ein künstliches neuronales Netz, was bei der maschinellen Verarbeitung von Bild- oder Audiodaten heutzutage oft verwendet wird. Als Datensatz wurde EMNIST verwendet, welcher in mehreren Datensatz-Varianten und Formaten vorliegt. Der EMNIST-Datensatz wurde in ein 28x28-Pixel-Bildformat abgeändert und in eine Datensatzstruktur konvertiert, die direkt mit dem MNIST-Datensatz übereinstimmt, weshalb der Name auch EMNIST lautet für *Extended MNIST*. Weiteres zum Datensatz ist auf der folgenden [Seite](https://www.nist.gov/itl/products-and-services/emnist-dataset) nachzulesen. Das Projekt ist im Bereich *Deep Learning* anzusiedeln. Deep Learning bezeichnet eine Methode des maschinellen Lernens, die eine Reihe hierarchischer Schichten bzw. eine Hierarchie von Konzepten nutzt, um den Prozess des maschinellen Lernens durchzuführen. Diese Schichten nennt man auch Zwischenschichten (englisch hidden layers), die zwischen Eingabeschicht und Ausgabeschicht eingesetzt werden um dadurch eine umfangreiche innere Struktur zu erzeugen. Es ist eine spezielle Methode der Informationsverarbeitung.

## Beschreibung

Das Projekt besteht wesentlich aus der *app.py*, welches als Tool dient um selbst geschriebene Ziffern auszuwerten. Zusätzlich sind weitere Pythonskripte vorhanden. *evalPicture.py* enthält die wichtigen Funktionen zum Auswerten eines Bildes nach unserem Format und *trainModel.py* enthält die Logik um ein CNN zu trainieren. Im Projekt wird ein bereits erstelltes CNN zur Verfügung gestellt namens *model_Balanced.h5*. Für das trainierte Model wurde von EMNIST das *Balanced DataSet* (131,600 Bilder | 47 balancierte Klassen) verwendet. Folgende Klassen sind dabei in Benutzung:
```python
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e',
'f', 'g', 'h', 'n', 'q', 'r', 't']
```
## trainModel.py
Im folgenden wird einmal der Aufbau des genannten Skriptes zum Trainieren eines Models erklärt. Als erstes wird der Datensatz importiert, der bereits zu 80% und 20% in TrainingSet und TestSet unterteilt wurde. Die Idee dahinter ist, dass man ein Overfitting durch das TrainingSet vermeiden möchte. Anders ausgedrückt ist das Problem, dass das Modell möglicherweise eine zu spezifische Funktion lernt, die mit den Trainingsdaten gut funktioniert, aber nicht auf Bilder die das Model noch nie gesehen hat. Der Datensatz wird durch folgende Aufrufe im Code gespeichert:
```python
training_images, training_labels = extract_training_samples('balanced')
test_images, test_labels = extract_test_samples('balanced')
```
Standardmäßig beträgt wie bereits gesagt die Form jedes Bildes im EMNIST-Datensatz 28 x 28, sodass wir nicht die Form aller Bilder überprüfen müssen. Wenn man reale Datensätze verwendet, hat man wahrscheinlich nicht so viel Glück. 28 x 28 ist auch eine ziemlich kleine Größe für ein Bild, sodass das CNN jedes Bild ziemlich schnell durchlaufen kann. Als Nächstes muss man den Datensatz umformen, die unser Modell beim Trainieren des Modells erwartet. Die erste Zahl ist die Anzahl der Bilder. Dann kommt die Form jedes Bildes (28x28). Die letzte Zahl ist 1, was bedeutet, dass die Bilder Graustufen sind.
```python
# reshape format [samples] [width] [height] [channels]
training_images = training_images.reshape(training_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
```
Im nächsten Schritt wird mit der Methode "One-Hot Encoding" die kategorischen Eigenschaften erzeugt. Kurz gesagt erzeugt diese Methode einen Vektor mit einer Länge gleich der Anzahl der Kategorien im Datensatz. Wenn ein Datenpunkt zu der i-ten Kategorie gehört, dann wird den Komponenten dieses Vektors der Wert 0 zugewiesen mit Ausnahme der i-ten Komponente, der den Wert 1 zugewiesen bekommt. Auf diese Weise kann man die Kategorien auf numerisch sinnvolle Weise verfolgen.
```python
training_labels = keras.utils.to_categorical(training_labels)
test_labels = keras.utils.to_categorical(test_labels)
```
Im nächsten Schritt des "Feature Engineerings" werden die Datensätze normalisiert, indem man durch 255 teilt. Man teilt durch 255, da die Pixel der Bilder als Graustufen gegeben sind, also einen Wert zwischen 0 und 255 haben und man möchte einen Wert zwischen 0 & 1 erhalten. Wenn man mit einem herkömmlichen CNN zur Bildklassifizierung arbeiten möchte, hat die Ausgabeschicht N Neuronen, wobei N die Anzahl der Bildklassen ist (hier 47), die man identifizieren möchte. Jedes Ausgabeneuron soll die Wahrscheinlichkeit darstellen, mit der man jede Bildklasse beobachtet hat. Somit eignet sich der Bereich zwischen 0 bis 1 gut zur Darstellung der Wahrscheinlichkeit.
```python
# normalize inputs
training_images = training_images / 255
test_images = test_images / 255
```
Zum Schluss ist man soweit das CNN zu bauen mit den verschiedenen Hidden-Layers und das fertige Modell abzusspeichern. Benutzt wurde zum Bauen des CNNs die Klassen und Methoden von *Keras*. Der Modelltyp, den wir verwenden, ist der sequentielle Modelltyp. Sequentiell ist der einfachste Weg, ein Modell in Keras zu erstellen. Es ermöglicht ein Modell Schicht für Schicht aufzubauen. Eine Schicht welche man oft benutzt ist die Conv2D Schicht von Keras. Um das ganze zu in Kürze zu verstehen kann man sich das folgende Beispiel auf der [Seite](https://cs231n.github.io/assets/conv-demo/index.html) anschauen. Folgendes kann man über diese Schicht sagen: Schichten am Anfang des CNN (d. h. näher am tatsächlichen Eingabebild) lernen weniger Faltungsfilter, während Schichten tiefer im Netzwerk (d. h. näher an den Ausgabe) mehr Filter lernen. Conv2D-Ebenen dazwischen lernen mehr Filter als die frühen Conv2D-Ebenen, aber weniger Filter als die Ebenen, die näher an der Ausgabe liegen. Eine weitere verwendete Schicht ist die MaxPooling2D-Schicht, die verwendet, um die räumlichen Dimensionen des Ausgabevolumens zu reduzieren. Eine gängige Praxis beim Entwerfen von CNN-Architekturen ist es, dass die Anzahl der gelernten Filter zunimmt wenn die räumlichen Ausgabevolumen abnehmen. Als Aktivierungsfunktion wird [ReLu](https://deepai.org/machine-learning-glossary-and-terms/relu) verwendet. Kurz gesagt löst ReLu das Problem der traditionellen Benutzung der Sigmoidfunktion und man spart Rechenzeit.  Um weiter Overfitting vorzubeugen werden sogenannte Dense-Layer verwendet. Während des Trainings wird eine bestimmte Anzahl von Layer-Ausgaben nach dem Zufallsprinzip ignoriert oder „ausgelassen“. Dies hat den Effekt, dass die Schicht wie eine Schicht mit einer anderen Anzahl von Knoten und Konnektivität zu der vorherigen Schicht aussieht und behandelt wird. Tatsächlich wird jede Aktualisierung einer Schicht während des Trainings mit einer anderen „Ansicht“ der konfigurierten Schicht durchgeführt. Als weitere wichtige Schicht kurz vor dem Schluss ist die Flattening-Schicht. Beim Flattening werden die Daten in ein 1-dimensionales Array umgewandelt, um sie in die nächste Schicht einzugeben. Wir glätten die Ausgabe der Faltungsschichten, um einen einzelnen langen Merkmalsvektor zu erstellen. Die letzte wichtige Layer ist die Dense-Layer, die man sich als Fully-Connected Layer vorstellen kann. Jedes Neuron in einer Schicht erhält einen Input von allen Neuronen, die in der vorherigen Schicht vorhanden sind – sie sind also dicht verbunden, weshalb man sie als Fully-Connected Layer bezeichnet.
Im folgenden sieht man Bilder, wie das ganze aussieht:
![alt text](https://github.com/kevingostomski/CNN/blob/main/doc/example1.jpg)
![alt text](https://github.com/kevingostomski/CNN/blob/main/doc/example2.jpg)
![alt text](https://github.com/kevingostomski/CNN/blob/main/doc/Result.jpeg)
![alt text](https://github.com/kevingostomski/CNN/blob/main/doc/ModelAccuracy.jpeg)
![alt text](https://github.com/kevingostomski/CNN/blob/main/doc/modelLoss.jpeg)

## Benutzung

_1_ = Benutzung kann über das Importen der *evalPicture.py* in ein eigenes Skript erfolgen.

_2_ = Ausführen der *app.py*





### Credits:

[Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters](http://arxiv.org/abs/1702.05373)
