# Die handschriftliche alphanumerische Erkennung
Sequential CNN for Handwritten Alphanumeric Recognition

[![Generic badge](https://img.shields.io/badge/ready-passing-<green>.svg)](https://shields.io/) ![Maintaner](https://img.shields.io/badge/maintainer-kevingostomski-blue)


## Überblick

Die Erkennung handgeschriebener alphanumerischer Ausdrücke ist eine schwierige Aufgabe für die Maschine, da handgeschriebene Ziffern nicht perfekt sind und auf vielen verschiedenen Weisen geschrieben werden können. Die handgeschriebene alphanumerische Ausdruckserkennung ist die Lösung für dieses Problem, die das Bild einer Ziffer verwendet um die im Bild vorhandene Ziffer zu erkennen. Das Projekt ist ein Pythonprogramm, welches die Ausdruckserkennung mithilfe eines *faltendem neuronalem Netzwerk (Englisch =  Convolutional Neural Network -> CNN) löst. Ein CNN ist ein künstliches neuronales Netz, was bei der maschinellen Verarbeitung von Bild- oder Audiodaten heutzutage oft verwendet wird. Als Datensatz wurde EMNIST verwendet, welcher in mehreren Datensatz-Varianten und Formaten vorliegt. Der EMNIST-Datensatz wurde in ein 28x28-Pixel-Bildformat abgeändert und in eine Datensatzstruktur konvertiert, die direkt mit dem MNIST-Datensatz übereinstimmt, weshalb der Name auch EMNIST lautet für *Extended MNIST*. Weiteres zum Datensatz ist auf der folgenden [Seite](https://www.nist.gov/itl/products-and-services/emnist-dataset) nachzulesen. Das Projekt ist im Bereich *Deep Learning* anzusiedeln. Deep Learning bezeichnet eine Methode des maschinellen Lernens, die eine Reihe hierarchischer Schichten bzw. eine Hierarchie von Konzepten nutzt, um den Prozess des maschinellen Lernens durchzuführen. Diese Schichten nennt man auch Zwischenschichten (englisch hidden layers), die zwischen Eingabeschicht und Ausgabeschicht eingesetzt werden um dadurch eine umfangreiche innere Struktur zu erzeugen. Es ist eine spezielle Methode der Informationsverarbeitung.
![grafik](https://github.com/miGa77/AMK-DataAnalysis/blob/main/doc/example1.jpg)
![grafik](https://github.com/miGa77/AMK-DataAnalysis/blob/main/doc/example2.jpg)

## Beschreibung

Das Projekt besteht wesentlich aus der *app.py*, welches als Tool dient um selbst geschriebene Ziffern auszuwerten. Zusätzlich sind weitere Pythonskripte vorhanden. *evalPicture.py* enthält die wichtigen Funktionen zum Auswerten eines Bildes nach unserem Format und *trainModel.py* enthält die Logik um ein CNN zu trainieren. Im Projekt sind zwei bereits erstellte CNNs zur Verfügung gestellt namens *model_Balanced.h5* und *trainedModel.h5*. Für das trainierte Model wurde von EMNIST das *Balanced DataSet* (131,600 Bilder | 47 balancierte Klassen) verwendet. Folgende Klassen sind dabei in Benutzung:
```python
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e',
'f', 'g', 'h', 'n', 'q', 'r', 't']
```
## trainModel.py
Im folgenden wird einmal der Aufbau des genannten Skriptes zum Trainieren eines Models erklärt. Das dargestellte Skript ist zum Erstellen des *trainedModel.h5*. Als erstes wird der Datensatz importiert, der bereits zu 80% und 20% in TrainingSet und TestSet unterteilt wurde. Die Idee dahinter ist, dass man ein Overfitting durch das TrainingSet vermeiden möchte. Anders ausgedrückt ist das Problem, dass das Modell möglicherweise eine zu spezifische Funktion lernt, die mit den Trainingsdaten gut funktioniert, aber nicht auf Bilder die das Model noch nie gesehen hat. Der Datensatz wird durch folgende Aufrufe im Code gespeichert:
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
Zum Schluss ist man soweit das CNN zu bauen mit den verschiedenen Hidden-Layers und das fertige Modell abzusspeichern. Benutzt wurde zum Bauen des CNNs die Klassen und Methoden von *Keras*. Der Modelltyp, den wir verwenden, ist der sequentielle Modelltyp. Sequentiell ist der einfachste Weg, ein Modell in Keras zu erstellen. Es ermöglicht ein Modell Schicht für Schicht aufzubauen. Eine Schicht welche man oft benutzt ist die Conv2D Schicht von Keras. Um das ganze zu verstehen kann man sich das folgende Beispiel auf der [Seite](https://cs231n.github.io/assets/conv-demo/index.html) anschauen. Folgendes kann man über diese Schicht sagen: Schichten am Anfang des CNN (d. h. näher am tatsächlichen Eingabebild) lernen weniger Faltungsfilter, während Schichten tiefer im Netzwerk (d. h. näher an den Ausgabe) mehr Filter lernen. Conv2D-Ebenen dazwischen lernen mehr Filter als die frühen Conv2D-Ebenen, aber weniger Filter als die Ebenen, die näher an der Ausgabe liegen. Eine weitere verwendete Schicht ist die MaxPooling2D-Schicht, die verwendet wird, um die räumlichen Dimensionen des Ausgabevolumens zu reduzieren. Eine gängige Praxis beim Entwerfen von CNN-Architekturen ist es, dass die Anzahl der gelernten Filter zunimmt wenn die räumlichen Ausgabevolumen abnehmen. Als Aktivierungsfunktion wird [ReLu](https://deepai.org/machine-learning-glossary-and-terms/relu) verwendet. Kurz gesagt löst ReLu das Problem der traditionellen Benutzung der Sigmoidfunktion und man spart Rechenzeit.  Um weiter Overfitting vorzubeugen werden sogenannte Dropout-Layer verwendet. Während des Trainings wird eine bestimmte Anzahl von Layer-Ausgaben nach dem Zufallsprinzip ignoriert oder „ausgelassen“. Dies hat den Effekt, dass die Schicht wie eine Schicht mit einer anderen Anzahl von Knoten und Konnektivität zu der vorherigen Schicht aussieht und behandelt wird. Tatsächlich wird jede Aktualisierung einer Schicht während des Trainings mit einer anderen „Ansicht“ der konfigurierten Schicht durchgeführt. Als weitere wichtige Schicht kurz vor dem Schluss ist die Flattening-Schicht. Beim Flattening werden die Daten in ein 1-dimensionales Array umgewandelt, um sie in die nächste Schicht einzugeben. Wir glätten die Ausgabe der Faltungsschichten, um einen einzelnen langen Merkmalsvektor zu erstellen. Die letzte wichtige Layer ist die Dense-Layer, die man sich als Fully-Connected Layer vorstellen kann. Jedes Neuron in einer Schicht erhält einen Input von allen Neuronen, die in der vorherigen Schicht vorhanden sind – sie sind also dicht verbunden, weshalb man sie als Fully-Connected Layer bezeichnet. Um sein Model zu erstellen und nun zu trainieren muss man folgende Zeilen ausführen:
```python
model = create_model()
history = model.fit(training_images, training_labels, validation_data=(test_images, test_labels), epochs=10,
                    batch_size=256, shuffle=True, verbose=2)
model.save('./data/save_trainedModel.h5')
```
Eine Epoche (epochs) gibt die Anzahl der Durchläufe des gesamten Trainingsdatensatzes an, die der maschinelle Lernalgorithmus abgeschlossen hat. Epochen kann man sich damit als Anzahl der Iterationen vorstellen. Die Batchgröße ist die Anzahl der Trainingsbeispiele, die man verwendet, um einen Schritt des stochastischen Gradientenabstiegs (SGD) durchzuführen.
Was ist SGD? SGD ist Gradient Descent (GD), aber anstatt das man alle Trainingsdaten auf einmal verwendet, um den Gradienten der Verlustfunktion in Bezug auf die Parameter des CNNs zu berechnen, verwendet man nur eine Teilmenge des Trainings-Datasets. Indem man nur eine Teilmenge der Trainingsdaten verwendet um den Gradienten zu berechnen, approximiert man den Gradienten stochastisch(dh man führt Rauschen(Noise) ein). Zur Auswertung schaut man sich die Lernkurven an. Dabei gibt es zwei Lernkurven, deren Verlauf sagen können, wie gut das trainierte Model letzen Endes ist. Die erste Lernkurve ist die Verlustkurve (loss curve). Das folgende Bild ist selbsterklärend, was man als gute Lernrate bezeichnen kann. 


![grafik](https://cs231n.github.io/assets/nn3/learningrates.jpeg)

Die wichtigere Kurve ist die Genauigkeit-Kurve (accuracy curve). 


![grafik](https://cs231n.github.io/assets/nn3/accuracies.jpeg)


Die Lücke zwischen Trainings- und Validierungsgenauigkeit ist ein klares Indiz wenn Overfitting vorliegt. Je größer die Lücke, desto höher ist das Overfitting.

### trainedModel.h5
___
![grafik](https://github.com/miGa77/AMK-DataAnalysis/blob/main/doc/1.jpg)
![grafik](https://github.com/miGa77/AMK-DataAnalysis/blob/main/doc/2.jpg)
![grafik](https://github.com/miGa77/AMK-DataAnalysis/blob/main/doc/3.jpg)
![grafik](https://github.com/miGa77/AMK-DataAnalysis/blob/main/doc/loss.jpg)
![grafik](https://github.com/miGa77/AMK-DataAnalysis/blob/main/doc/accuracy.jpg)

### model_Balanced.h5
___
![grafik](https://github.com/miGa77/AMK-DataAnalysis/blob/main/doc/Result.jpeg)
![grafik](https://github.com/miGa77/AMK-DataAnalysis/blob/main/doc/ModelAccuracy.jpeg)
![grafik](https://github.com/miGa77/AMK-DataAnalysis/blob/main/doc/modelLoss.jpeg)

## Tuner.py
Das *Tuner.py* Skript erstellt ebenfalls ein Model. Dabei wurde die Variante eines automatischen Hyperparameter-Tuning durchgeführt. Kurz gesagt dienen Hyperparameter zur Steuerung des Trainingsalgorithmus und dessen Wert im Gegensatz zu anderen Parametern vor dem eigentlichen Training des Modells festgelegt werden muss. Dazu gehört zum Beipsiel die Größe des Kernels bei einem Conv2d-Layer oder der Wert wie groß ein Dropout sein muss. Das Skript dient zum Veranschaulichen und Testen von Parametern. Zum Ausführen benötigt man viel Zeit. Das Resultat und Ergebnis ist die Feststellung die bereits im Abschnitt *trainModel.py* genannt wurde.

## Benutzung

_1_ = Benutzung kann über das Importen der *evalPicture.py* in ein eigenes Skript erfolgen.

_2_ = Ausführen der *app.py*

## Probleme & Lösung
* Bilder sind zentriert und haben kein Noise da der Datenbestand bereits gereinigt zur Verfügung gestellt wurde
&rightarrow; Lösung ist das Benutzen im *trainModel.py* Skript eines ImageDataGenerators. Dieser kann die Bilder des DataSets nehmen und zufällige Bilder verändern anhand von angegebenen Parametern. Wichtig zu erwähnen ist, das keine Bilder generiert und den Datenbestand hinzugefügt wird, es werden Bilder im Datenbestand verändert
* Probleme beim Benutzen von unbalancierten Klassen, die aber prinzipiell einen größeren Datenbestand haben
&rightarrow; Lösung ist das Benutzen von balancierten Klassen. Es ist durchaus üblich, dass in einem Dataset eine unausgeglichene Klassenverteilung vorliegt. Um dieses Problem zu lösen, stehen zwei gängige Methoden zur Verfügung, die Oversampling und Undersampling genannt werden. EMNIST hat bereits das Problem von unbalancierten Klassen gelöst und bietet balancierte Klassen an.
* Probleme beim Auswerten von Bildern wo zu nah am Rand geschrieben wurde
&rightarrow; Lösung ist das Schreiben einer Funktion, die das eigentlichte Bild nimmt und mittig auf ein größeres Bild packt. Nicht mit Resizen zu verwechseln.
Problem war das ROI (Region of Interest) in einer Funktion 0 war, was nicht vorkommen darf.
* Problem das wenn man einen Buchstaben auf das ganze Canvas-Board zeichnet, der Stift zu dünn ist
&rightarrow; Button zum Handlen von der Stiftdicke
* Beim Ausschneiden des Canvas-Boards kam ein graue Umrandung mit
&rightarrow; Auschneiden anpassen und verbessern 
* Immer noch Probleme beim Auswerten von bestimmten Ziffern [1,I] [2,Z] [9,g,q] [F,f] 
 &rightarrow; Lösung wäre höchstwahrscheinlich durch ein Oversampling, was zeitlich nicht hinhaut, da man das Model mithilfe der CSV erstellen muss. Zusätzlich müsste man weitere Datenbestände von den "Problemziffern" suchen und mergen oder selbst weitere erzeugen und hinzufügen Im folgendem Bild ist das Problem einmal sichtbar.
 
![grafik](https://github.com/miGa77/AMK-DataAnalysis/blob/main/doc/test_2.jpg)
 
 * Weiteres Problem im folgendem Bild sichtbar. Lösung ist keine vorhanden
![grafik](https://github.com/miGa77/AMK-DataAnalysis/blob/main/doc/Bild.jpg)
 
 
### Credits:

[Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters](http://arxiv.org/abs/1702.05373)
