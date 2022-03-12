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


class SegTableWidget(TableWidget):
    def __init__(self, parent=None):
        super(SegTableWidget, self).__init__(parent=parent)


class GrayscaleTableWidget(TableWidget):
    def __init__(self, parent=None):
        super(GrayscaleTableWidget, self).__init__(parent=parent)

        self.kgray_spinBox = QSpinBox()
        self.kgray_spinBox.setFixedSize(80, 40)
        self.kgray_spinBox.setObjectName('kgray')
        self.kgray_spinBox.setMinimum(-1000)
        self.kgray_spinBox.setMaximum(1000)
        self.kgray_spinBox.setValue(0)
        self.kgray_spinBox.setSingleStep(50)


        self.setColumnCount(2)
        self.setRowCount(1)
        self.setColumnWidth(0, 80)
        self.setColumnWidth(1, 80)
        self.setItem(0, 0, QTableWidgetItem('灰度偏移'))
        self.setCellWidget(0, 1, self.kgray_spinBox)

        self.signal_connect()

class ContrastTableWidget(TableWidget):
    def __init__(self, parent=None):
        super(ContrastTableWidget, self).__init__(parent=parent)
        self.kcontrast_spinBox = QSpinBox()
        self.kcontrast_spinBox.setFixedSize(80, 40)
        self.kcontrast_spinBox.setObjectName('kcontrast')
        self.kcontrast_spinBox.setMinimum(1)
        self.kcontrast_spinBox.setMaximum(10)
        self.kcontrast_spinBox.setValue(1)
        self.kcontrast_spinBox.setSingleStep(1)

        self.setColumnCount(2)
        self.setRowCount(1)
        self.setColumnWidth(0, 100)
        self.setColumnWidth(1, 80)
        self.setItem(0, 0, QTableWidgetItem('对比度增强'))
        self.setCellWidget(0, 1, self.kcontrast_spinBox)

        self.signal_connect()

class CutTableWidget(TableWidget):
    def __init__(self, parent=None):
        super(CutTableWidget, self).__init__(parent=parent)
        self.kfor_spinBox = QSpinBox()
        self.kfor_spinBox.setFixedSize(60, 40)
        self.kfor_spinBox.setObjectName('kfor')
        self.kfor_spinBox.setMinimum(0)
        self.kfor_spinBox.setMaximum(159)
        self.kfor_spinBox.setValue(159)
        self.kfor_spinBox.setSingleStep(1)

        self.kback_spinBox = QSpinBox()
        self.kback_spinBox.setObjectName('kback')
        self.kback_spinBox.setMinimum(0)
        self.kback_spinBox.setMaximum(159)
        self.kback_spinBox.setValue(0)
        self.kback_spinBox.setSingleStep(1)

        self.kup_spinBox = QSpinBox()
        self.kup_spinBox.setFixedSize(60, 40)
        self.kup_spinBox.setObjectName('kup')
        self.kup_spinBox.setMinimum(0)
        self.kup_spinBox.setMaximum(199)
        self.kup_spinBox.setValue(199)
        self.kup_spinBox.setSingleStep(1)

        self.kdown_spinBox = QSpinBox()
        self.kdown_spinBox.setObjectName('kdown')
        self.kdown_spinBox.setMinimum(0)
        self.kdown_spinBox.setMaximum(199)
        self.kdown_spinBox.setValue(0)
        self.kdown_spinBox.setSingleStep(1)

        self.kleft_spinBox = QSpinBox()
        self.kleft_spinBox.setObjectName('kleft')
        self.kleft_spinBox.setMinimum(0)
        self.kleft_spinBox.setMaximum(199)
        self.kleft_spinBox.setValue(0)
        self.kleft_spinBox.setSingleStep(1)

        self.kright_spinBox = QSpinBox()
        self.kright_spinBox.setFixedSize(60, 40)
        self.kright_spinBox.setObjectName('kright')
        self.kright_spinBox.setMinimum(0)
        self.kright_spinBox.setMaximum(199)
        self.kright_spinBox.setValue(159)
        self.kright_spinBox.setSingleStep(1)

        self.setColumnCount(4)
        self.setRowCount(3)
        self.setColumnWidth(0, 80)
        self.setColumnWidth(1, 60)
        self.setColumnWidth(2, 5)
        self.setColumnWidth(3, 60)
        self.setItem(0, 0, QTableWidgetItem('X轴范围'))
        self.setCellWidget(0, 1, self.kleft_spinBox)
        self.setItem(0, 2, QTableWidgetItem('-'))
        self.setCellWidget(0, 3, self.kright_spinBox)

        self.setItem(1, 0, QTableWidgetItem('Y轴范围'))
        self.setCellWidget(1, 1, self.kback_spinBox)
        self.setItem(1, 2, QTableWidgetItem('-'))
        self.setCellWidget(1, 3, self.kfor_spinBox)

        self.setItem(2, 0, QTableWidgetItem('Z轴范围'))
        self.setCellWidget(2, 1, self.kdown_spinBox)
        self.setItem(2, 2, QTableWidgetItem('-'))
        self.setCellWidget(2, 3, self.kup_spinBox)

        self.signal_connect()

class SaveTableWidget(TableWidget):
    def __init__(self, parent=None):
        super(SaveTableWidget, self).__init__(parent=parent)
