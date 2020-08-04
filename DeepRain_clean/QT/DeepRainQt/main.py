from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QIcon, QPixmap
import sys
FILE = "/home/simon/Dokumente/Teamprojekt/DeepRain_clean/Data/test/736x864_366x494/0801050950.png"

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        ui = uic.loadUi('form.ui', self)

        self.true_img = self.findChild(QtWidgets.QGraphicsView,"graphicsView_2")



        self.scene = QtWidgets.QGraphicsScene()
        self.pixmap = QPixmap(FILE)

        self.scene.addPixmap(self.pixmap)
        self.true_img.setScene(self.scene)
        self.true_img.show()
        self.show()




app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()