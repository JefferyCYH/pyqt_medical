from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class TableWidget(QTableWidget):
    def __init__(self, parent=None):
        super(TableWidget, self).__init__(parent=parent)
        self.mainwindow = parent
        self.setShowGrid(True)  # 显示网格
        self.setAlternatingRowColors(True)  # 隔行显示颜色
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.horizontalHeader().setVisible(False)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().sectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().sectionResizeMode(QHeaderView.Stretch)
        self.horizontalHeader().setStretchLastSection(True)
        self.setFocusPolicy(Qt.NoFocus)

    def signal_connect(self):
        for spinbox in self.findChildren(QSpinBox):
            spinbox.valueChanged.connect(self.update_item)
        for doublespinbox in self.findChildren(QDoubleSpinBox):
            doublespinbox.valueChanged.connect(self.update_item)
        for combox in self.findChildren(QComboBox):
            combox.currentIndexChanged.connect(self.update_item)
        for checkbox in self.findChildren(QCheckBox):
            checkbox.stateChanged.connect(self.update_item)

    def update_item(self):
        param = self.get_params()
        self.mainwindow.useListWidget.currentItem().update_params(param)
        self.mainwindow.update_image()

    def update_params(self, param=None):
        for key in param.keys():
            box = self.findChild(QWidget, name=key)
            if isinstance(box, QSpinBox) or isinstance(box, QDoubleSpinBox):
                box.setValue(param[key])
            elif isinstance(box, QComboBox):
                box.setCurrentIndex(param[key])
            elif isinstance(box, QCheckBox):
                box.setChecked(param[key])

    def get_params(self):
        param = {}
        for spinbox in self.findChildren(QSpinBox):
            param[spinbox.objectName()] = spinbox.value()
        for doublespinbox in self.findChildren(QDoubleSpinBox):
            param[doublespinbox.objectName()] = doublespinbox.value()
        for combox in self.findChildren(QComboBox):
            param[combox.objectName()] = combox.currentIndex()
        for combox in self.findChildren(QCheckBox):
            param[combox.objectName()] = combox.isChecked()
        return param


class WidthTableWidget(TableWidget):
    def __init__(self, parent=None):
        super(LightTableWidget, self).__init__(parent=parent)

        self.alpha_spinBox = QDoubleSpinBox()
        self.alpha_spinBox.setMinimum(0)
        self.alpha_spinBox.setMaximum(3)
        self.alpha_spinBox.setSingleStep(0.1)
        self.alpha_spinBox.setObjectName('alpha')

        self.beta_spinbox = QSpinBox()
        self.beta_spinbox.setMinimum(0)
        self.beta_spinbox.setSingleStep(1)
        self.beta_spinbox.setObjectName('beta')

        self.setColumnCount(2)
        self.setRowCount(2)

        self.setItem(0, 0, QTableWidgetItem('alpha'))
        self.setCellWidget(0, 1, self.alpha_spinBox)
        self.setItem(1, 0, QTableWidgetItem('beta'))
        self.setCellWidget(1, 1, self.beta_spinbox)
        self.signal_connect()


class GammaITabelWidget(TableWidget):
    def __init__(self, parent=None):
        super(GammaITabelWidget, self).__init__(parent=parent)
        self.gamma_spinbox = QDoubleSpinBox()
        self.gamma_spinbox.setMinimum(0)
        self.gamma_spinbox.setSingleStep(0.1)
        self.gamma_spinbox.setObjectName('gamma')

        self.setColumnCount(2)
        self.setRowCount(1)

        self.setItem(0, 0, QTableWidgetItem('gamma'))
        self.setCellWidget(0, 1, self.gamma_spinbox)
        self.signal_connect()
