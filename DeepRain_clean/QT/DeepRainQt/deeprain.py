# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'form.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
FILE = "/home/simon/Dokumente/Teamprojekt/DeepRain_clean/Data/train/736x864_366x494/1012311800.png"



class Ui_eval(object):
    def setupUi(self, eval):
        eval.setObjectName("eval")
        eval.resize(1150, 674)
        self.tabWidget = QtWidgets.QTabWidget(eval)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 1151, 671))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.horizontalSlider = QtWidgets.QSlider(self.tab)
        self.horizontalSlider.setGeometry(QtCore.QRect(40, 510, 781, 21))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.label = QtWidgets.QLabel(self.tab)
        self.label.setGeometry(QtCore.QRect(664, 260, 81, 20))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.tab)
        self.label_2.setGeometry(QtCore.QRect(664, 20, 81, 20))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.tab)
        self.label_3.setGeometry(QtCore.QRect(660, 50, 201, 161))
        self.label_3.setText("")
        self.label_3.setPixmap(QtGui.QPixmap("/home/simon/Dokumente/Teamprojekt/DeepRain_clean/Data/train/736x864_366x494/1012311800.png"))
        self.label_3.setScaledContents(True)
        self.label_3.setObjectName("label_3")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tabWidget.addTab(self.tab_2, "")

        self.retranslateUi(eval)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(eval)

    def retranslateUi(self, eval):
        _translate = QtCore.QCoreApplication.translate
        eval.setWindowTitle(_translate("eval", "eval"))
        self.label.setText(_translate("eval", "Prediction"))
        self.label_2.setText(_translate("eval", "Label"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("eval", "Tab 1"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("eval", "Tab 2"))



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    eval = QtWidgets.QWidget()
    ui = Ui_eval()
    ui.setupUi(eval)
    eval.show()
    sys.exit(app.exec_())

