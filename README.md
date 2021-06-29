# DataAnalysis

| Participant    | Identifier       |
| ------------- |:-------------:|
| Kevin Gostomski     | kg3515s|
| Michael Gatterdam      |  mg9938s    |
| André Eichstaedt | ae2894s      |

## Einleitung

### Die handschriftliche alphanumerische Erkennung

Die Erkennung handgeschriebener alphanumerischer Ausdrücke ist eine schwierige Aufgabe für die Maschine, da handgeschriebene Ziffern nicht perfekt sind und auf vielen verschiedenen Weisen geschrieben werden können. Die handgeschriebene alphanumerische Ausdruckserkennung ist die Lösung für dieses Problem, die das Bild einer Ziffer verwendet um die im Bild vorhandene Ziffer zu erkennen. Das Projekt ist ein Pythonprogramm, welches die Ausdruckserkennung mithilfe eines *faltendem neuronalem Netzwerk*
(Englisch =  Convolutional Neural Network -> CNN) löst. Ein CNN ist ein künstliches neuronales Netz, was bei der maschinellen Verarbeitung von Bild- oder Audiodaten heutzutage oft verwendet wird. Als Datensatz wurde EMNIST verwendet, welcher in mehreren Datensatz-Varianten und Formaten vorliegt. Der EMNIST-Datensatz wurde in ein 28x28-Pixel-Bildformat abgeändert und in eine Datensatzstruktur konvertiert, die direkt mit dem MNIST-Datensatz übereinstimmt, weshalb der Name auch EMNIST lautet für *Extended MNIST*. Weiteres zum Datensatz ist auf der folgenden [Seite](https://www.nist.gov/itl/products-and-services/emnist-dataset) nachzulesen.

*Subject area: Machine Learning, Deep Learning, Neural Network Architectures*
