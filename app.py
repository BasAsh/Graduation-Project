from PyQt5 import QtCore, QtWidgets, QtGui
import sys
import os
import sqlite3

from eyeTraking import start_test
from gazeheatplot import gazeheatplot
from gaze_tracking_frame import gaze_tracking_frame

import cv2
import tensorflow as tf
import os
import cv2
import numpy as np


db_path = os.path.abspath(os.path.join('.', 'child_data.db'))
db_conn = sqlite3.connect(db_path, check_same_thread=False)

model_path = os.path.abspath(os.path.join('.', 'fmodel.h5'))
model = tf.keras.models.load_model(model_path, compile=False)


def preprocessing_img(img_path):
    img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (48, 48))
    img = np.array(img, dtype='float').reshape(-1, 48, 48, 3)
    img = np.array(img, dtype='float') / 250.0
    return img


def predict_autism():
    img1_path = os.path.abspath(os.path.join(
        '.', 'ASD_heatmaps', 'ASD_scanpath_2.png'))
    img2_path = os.path.abspath(os.path.join(
        '.', 'TD_heatmaps', 'TD_scanpath_39.png'))

    img1 = preprocessing_img(img1_path)
    img2 = preprocessing_img(img2_path)
    pred_idxs = model.predict(img1)
    pred_idxs = np.argmax(pred_idxs, axis=1)[0]
    if pred_idxs == 0:
        res = 'You have been diagnosed with AUTISM'
    else:
        res = 'You have been diagnosed to be a non autistic child'
    return res


# Welcome page
class WelcomePage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(WelcomePage, self).__init__(parent)
        self.setupUI()
        self.show()

    def setupUI(self):
        self.title1 = QtWidgets.QLabel('FAMELIA System')
        self.title1.setFont(QtGui.QFont('Arial', 24))
        self.title1.setAlignment(QtCore.Qt.AlignCenter)

        self.title2 = QtWidgets.QLabel('Autisem Diagnosis')
        self.title2.setFont(QtGui.QFont('Arial', 20))
        self.title2.setAlignment(QtCore.Qt.AlignCenter)

        self.img = QtWidgets.QLabel('')
        self.img.setAlignment(QtCore.Qt.AlignCenter)
        self.pixmap = QtGui.QPixmap('profile.jpeg')
        self.pixmap = self.pixmap.scaled(600, 315)
        self.img.setPixmap(self.pixmap)

        self.start = QtWidgets.QPushButton('Start')
        self.start.setFont(QtGui.QFont('Arial', 14))
        self.start.resize(self.start.minimumSizeHint())

        vbox_layout = QtWidgets.QVBoxLayout()
        self.setLayout(vbox_layout)

        vbox_layout.addWidget(self.title1)
        vbox_layout.addWidget(self.title2)
        vbox_layout.addWidget(self.img)
        vbox_layout.addWidget(self.start)


# Login page
class LoginPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(LoginPage, self).__init__(parent)
        self.setupUI()
        self.show()

    def setupUI(self):
        self.title = QtWidgets.QLabel('Sign-in')
        self.title.setFont(QtGui.QFont('Arial', 20))
        self.title.setAlignment(QtCore.Qt.AlignCenter)

        self.email = QtWidgets.QLabel('Email')
        self.email.setFont(QtGui.QFont('Arial', 14))
        self.email.setAlignment(QtCore.Qt.AlignLeft)

        self.text_edit_email = QtWidgets.QLineEdit()
        self.text_edit_email.setFont(QtGui.QFont('Arial', 14))
        self.text_edit_email.setAlignment(QtCore.Qt.AlignLeft)

        self.password = QtWidgets.QLabel('Password')
        self.password.setFont(QtGui.QFont('Arial', 14))
        self.password.setAlignment(QtCore.Qt.AlignLeft)

        self.text_edit_password = QtWidgets.QLineEdit()
        self.text_edit_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.text_edit_password.setFont(QtGui.QFont('Arial', 14))
        self.text_edit_password.setAlignment(QtCore.Qt.AlignLeft)

        self.msg = QtWidgets.QLabel('New user? Sign up')
        self.msg.setFont(QtGui.QFont('Arial', 12))
        self.msg.setAlignment(QtCore.Qt.AlignLeft)

        self.warning = QtWidgets.QLabel('Fill the missing data')
        self.warning.setFont(QtGui.QFont('Arial', 12))
        self.warning.setAlignment(QtCore.Qt.AlignLeft)
        self.warning.setStyleSheet('color: red')
        self.warning.setVisible(False)

        self.wrong_data = QtWidgets.QLabel('Invalid Email or Password')
        self.wrong_data.setFont(QtGui.QFont('Arial', 12))
        self.wrong_data.setAlignment(QtCore.Qt.AlignLeft)
        self.wrong_data.setStyleSheet('color: red')
        self.wrong_data.setVisible(False)
        self.wrong_data.setVisible(False)

        self.login = QtWidgets.QPushButton('Login')
        self.login.setFont(QtGui.QFont('Arial', 14))
        self.login.resize(self.login.minimumSizeHint())

        self.signup = QtWidgets.QPushButton('Here')
        self.signup.setFont(QtGui.QFont('Arial', 12))
        self.signup.resize(self.signup.minimumSizeHint())

        grid_layout = QtWidgets.QGridLayout()
        self.setLayout(grid_layout)

        grid_layout.addWidget(QtWidgets.QLabel(''), 0, 0, 1, 3)
        grid_layout.addWidget(self.title, 1, 1)
        grid_layout.addWidget(QtWidgets.QLabel(''), 2, 0, 1, 3)
        grid_layout.addWidget(self.email, 3, 0)
        grid_layout.addWidget(self.text_edit_email, 3, 1)
        grid_layout.addWidget(self.password, 4, 0)
        grid_layout.addWidget(self.text_edit_password, 4, 1)
        grid_layout.addWidget(QtWidgets.QLabel(''), 5, 0, 1, 3)
        grid_layout.addWidget(self.msg, 6, 0)
        grid_layout.addWidget(self.signup, 6, 1)
        grid_layout.addWidget(self.warning, 7, 1)
        grid_layout.addWidget(self.wrong_data, 8, 1)
        grid_layout.addWidget(QtWidgets.QLabel(''), 9, 0, 1, 3)
        grid_layout.addWidget(self.login, 10, 1)
        grid_layout.addWidget(QtWidgets.QLabel(''), 11, 0, 1, 3)


# SignUp page
class SignUpPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(SignUpPage, self).__init__(parent)
        self.gender = 'Male'
        self.agree = False
        self.setupUI()
        self.show()

    def is_male(self, selected):
        if selected:
            self.gender = 'Male'

    def is_female(self, selected):
        if selected:
            self.gender = 'Female'

    def is_agreed(self):
        if self.agree_chb.isChecked():
            self.agree = True

    def setupUI(self):
        self.title = QtWidgets.QLabel('Sign-up')
        self.title.setFont(QtGui.QFont('Arial', 20))
        self.title.setAlignment(QtCore.Qt.AlignCenter)

        self.fullname = QtWidgets.QLabel('Full Name')
        self.fullname.setFont(QtGui.QFont('Arial', 12))
        self.fullname.setAlignment(QtCore.Qt.AlignLeft)

        self.text_edit_fullname = QtWidgets.QLineEdit()
        self.text_edit_fullname.setFont(QtGui.QFont('Arial', 12))
        self.text_edit_fullname.setAlignment(QtCore.Qt.AlignLeft)

        self.email = QtWidgets.QLabel('Email')
        self.email.setFont(QtGui.QFont('Arial', 12))
        self.email.setAlignment(QtCore.Qt.AlignLeft)

        self.text_edit_email = QtWidgets.QLineEdit()
        self.text_edit_email.setFont(QtGui.QFont('Arial', 12))
        self.text_edit_email.setAlignment(QtCore.Qt.AlignLeft)

        self.password = QtWidgets.QLabel('Password')
        self.password.setFont(QtGui.QFont('Arial', 12))
        self.password.setAlignment(QtCore.Qt.AlignLeft)

        self.text_edit_password = QtWidgets.QLineEdit()
        self.text_edit_password.setFont(QtGui.QFont('Arial', 12))
        self.text_edit_password.setAlignment(QtCore.Qt.AlignLeft)

        self.text_edit_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.text_edit_password.setFont(QtGui.QFont('Arial', 12))
        self.text_edit_password.setAlignment(QtCore.Qt.AlignLeft)

        self.age = QtWidgets.QLabel('Age')
        self.age.setFont(QtGui.QFont('Arial', 12))
        self.age.setAlignment(QtCore.Qt.AlignLeft)

        self.text_edit_age = QtWidgets.QLineEdit()
        self.text_edit_age.setMaxLength(3)
        self.int_validator = QtGui.QRegExpValidator(QtCore.QRegExp('\d+'))
        self.text_edit_age.setValidator(self.int_validator)
        self.text_edit_age.setFont(QtGui.QFont('Arial', 12))
        self.text_edit_age.setAlignment(QtCore.Qt.AlignLeft)

        self.gender_lbl = QtWidgets.QLabel('Gender')
        self.gender_lbl.setFont(QtGui.QFont('Arial', 12))
        self.gender_lbl.setAlignment(QtCore.Qt.AlignLeft)

        self.rbtn_male = QtWidgets.QRadioButton('Male')
        self.rbtn_male.setFont(QtGui.QFont('Arial', 12))
        self.rbtn_male.setChecked(True)
        self.rbtn_male.toggled.connect(self.is_male)

        self.rbtn_female = QtWidgets.QRadioButton('Female')
        self.rbtn_female.setFont(QtGui.QFont('Arial', 12))
        self.rbtn_female.toggled.connect(self.is_female)
        self.rbtn_female.setChecked(False)

        self.agree_chb = QtWidgets.QCheckBox('Agree to terms and polices')
        self.agree_chb.toggled.connect(self.is_agreed)
        self.agree_chb.setFont(QtGui.QFont('Arial', 12))

        self.submit = QtWidgets.QPushButton('Submit')
        self.submit.setFont(QtGui.QFont('Arial', 12))

        self.warning = QtWidgets.QLabel('Fill the missing data')
        self.warning.setFont(QtGui.QFont('Arial', 12))
        self.warning.setAlignment(QtCore.Qt.AlignLeft)
        self.warning.setStyleSheet('color: red')
        self.warning.setVisible(False)

        grid_layout = QtWidgets.QGridLayout()
        self.setLayout(grid_layout)

        grid_layout.addWidget(QtWidgets.QLabel(''), 0, 0, 1, 3)
        grid_layout.addWidget(self.title, 1, 1)
        grid_layout.addWidget(QtWidgets.QLabel(''), 2, 0, 1, 3)

        grid_layout.addWidget(self.fullname, 3, 0)
        grid_layout.addWidget(self.text_edit_fullname, 3, 1)

        grid_layout.addWidget(self.email, 4, 0)
        grid_layout.addWidget(self.text_edit_email, 4, 1)

        grid_layout.addWidget(self.password, 5, 0)
        grid_layout.addWidget(self.text_edit_password, 5, 1)

        grid_layout.addWidget(self.age, 6, 0)
        grid_layout.addWidget(self.text_edit_age, 6, 1)

        grid_layout.addWidget(self.gender_lbl, 7, 0)
        grid_layout.addWidget(self.rbtn_male, 7, 1)
        grid_layout.addWidget(self.rbtn_female, 7, 2)

        grid_layout.addWidget(QtWidgets.QLabel(''), 8, 0, 1, 3)
        grid_layout.addWidget(self.agree_chb, 9, 1)
        grid_layout.addWidget(self.warning, 10, 1)
        grid_layout.addWidget(self.submit, 11, 1)
        grid_layout.addWidget(QtWidgets.QLabel(''), 12, 0, 1, 3)


# ChildData page
class ChildDataPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ChildDataPage, self).__init__(parent)
        self.gender = 'Male'
        self.setupUI()
        self.show()

    def is_male(self, selected):
        if selected:
            self.gender = 'Male'

    def is_female(self, selected):
        if selected:
            self.gender = 'Female'

    def setupUI(self):
        self.title = QtWidgets.QLabel('Child Data')
        self.title.setFont(QtGui.QFont('Arial', 20))
        self.title.setAlignment(QtCore.Qt.AlignCenter)

        self.name = QtWidgets.QLabel('Child Name')
        self.name.setFont(QtGui.QFont('Arial', 12))
        self.name.setAlignment(QtCore.Qt.AlignCenter)

        self.text_edit_name = QtWidgets.QLineEdit()
        self.text_edit_name.setFont(QtGui.QFont('Arial', 12))
        self.text_edit_name.setAlignment(QtCore.Qt.AlignCenter)

        self.age = QtWidgets.QLabel('Age')
        self.age.setFont(QtGui.QFont('Arial', 12))
        self.age.setAlignment(QtCore.Qt.AlignCenter)

        self.text_edit_age = QtWidgets.QLineEdit()
        self.text_edit_age.setMaxLength(3)
        self.int_validator = QtGui.QRegExpValidator(QtCore.QRegExp('\d+'))
        self.text_edit_age.setValidator(self.int_validator)
        self.text_edit_age.setFont(QtGui.QFont('Arial', 12))
        self.text_edit_age.setAlignment(QtCore.Qt.AlignCenter)

        self.gender_lbl = QtWidgets.QLabel('Gender')
        self.gender_lbl.setFont(QtGui.QFont('Arial', 12))
        self.gender_lbl.setAlignment(QtCore.Qt.AlignCenter)

        self.rbtn_male = QtWidgets.QRadioButton('Male')
        self.rbtn_male.setFont(QtGui.QFont('Arial', 12))
        self.rbtn_male.setChecked(True)
        self.rbtn_male.toggled.connect(self.is_male)

        self.rbtn_female = QtWidgets.QRadioButton('Female')
        self.rbtn_female.setFont(QtGui.QFont('Arial', 12))
        self.rbtn_female.setChecked(False)
        self.rbtn_female.toggled.connect(self.is_female)

        self.warning = QtWidgets.QLabel('Fill the missing data')
        self.warning.setFont(QtGui.QFont('Arial', 12))
        self.warning.setAlignment(QtCore.Qt.AlignCenter)
        self.warning.setStyleSheet('color: red')
        self.warning.setVisible(False)

        self.success = QtWidgets.QLabel('Your data has been submitted')
        self.success.setFont(QtGui.QFont('Arial', 12))
        self.success.setAlignment(QtCore.Qt.AlignCenter)
        self.success.setStyleSheet('color: green')
        self.success.setVisible(False)

        self.wrong_age = QtWidgets.QLabel('Invalid child age')
        self.wrong_age.setFont(QtGui.QFont('Arial', 12))
        self.wrong_age.setAlignment(QtCore.Qt.AlignCenter)
        self.wrong_age.setStyleSheet('color: red')
        self.wrong_age.setVisible(False)

        self.submit = QtWidgets.QPushButton('Submit')
        self.submit.setFont(QtGui.QFont('Arial', 12))
        self.submit.resize(self.submit.minimumSizeHint())

        self.start_test = QtWidgets.QPushButton('Start Test')
        self.start_test.setFont(QtGui.QFont('Arial', 12))
        self.start_test.resize(self.start_test.minimumSizeHint())

        self.cvt_sp_hm = QtWidgets.QPushButton('Convert Scanpath to Heatmap')
        self.cvt_sp_hm.setFont(QtGui.QFont('Arial', 12))
        self.cvt_sp_hm.resize(self.cvt_sp_hm.minimumSizeHint())

        self.show_result = QtWidgets.QPushButton('Show Result')
        self.show_result.setFont(QtGui.QFont('Arial', 12))
        self.show_result.resize(self.show_result.minimumSizeHint())

        self.video_image = QtWidgets.QPushButton('Video Image')
        self.video_image.setFont(QtGui.QFont('Arial', 12))
        self.video_image.resize(self.video_image.minimumSizeHint())

        grid_layout = QtWidgets.QGridLayout()
        self.setLayout(grid_layout)

        grid_layout.addWidget(QtWidgets.QLabel(''), 0, 0, 1, 3)
        grid_layout.addWidget(self.title, 1, 1)
        grid_layout.addWidget(QtWidgets.QLabel(''), 2, 0, 1, 3)

        grid_layout.addWidget(self.name, 3, 0)
        grid_layout.addWidget(self.text_edit_name, 3, 1)

        grid_layout.addWidget(self.age, 4, 0)
        grid_layout.addWidget(self.text_edit_age, 4, 1)

        grid_layout.addWidget(self.warning, 5, 1)
        grid_layout.addWidget(self.success, 6, 1)
        grid_layout.addWidget(self.wrong_age, 7, 1)

        grid_layout.addWidget(self.gender_lbl, 8, 0)
        grid_layout.addWidget(self.rbtn_male, 8, 1)
        grid_layout.addWidget(self.rbtn_female, 8, 2)
        grid_layout.addWidget(QtWidgets.QLabel(''), 11, 0, 1, 3)

        grid_layout.addWidget(self.submit, 12, 1)
        grid_layout.addWidget(self.start_test, 13, 1)
        grid_layout.addWidget(self.cvt_sp_hm, 14, 1)
        grid_layout.addWidget(self.show_result, 15, 1)
        grid_layout.addWidget(self.video_image, 16, 1)
        grid_layout.addWidget(QtWidgets.QLabel(''), 17, 1, 1, 3)


# Welcome -> Signin -> Signup -> Signin -> ChildData
# Welcome -> Signin -> ChildData

# Main window
class MainWindow(QtWidgets.QMainWindow):
    # Create main window
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setStyleSheet('background-color: #cbc4cd')
        self.setupUI()
        self.show()

    def setupUI(self):
        self.show_welcome_page()

    def show_welcome_page(self):
        self.welcome_page = WelcomePage(self)
        self.setWindowTitle('Welcome')
        self.setCentralWidget(self.welcome_page)
        self.welcome_page.start.clicked.connect(self.show_login_page)

    def show_login_page(self):
        self.login_page = LoginPage(self)
        self.setWindowTitle('Login')
        self.setCentralWidget(self.login_page)
        self.login_page.login.clicked.connect(self.logging_in)
        self.login_page.signup.clicked.connect(self.show_signup_page)

    def logging_in(self):
        email = self.login_page.text_edit_email.text().lower()
        pw = self.login_page.text_edit_password.text()
        if email == '' or pw == '':
            self.login_page.warning.setVisible(True)
            self.login_page.wrong_data.setVisible(False)
        else:
            query = ''' SELECT * FROM user_details WHERE Email = ? AND Password = ? '''
            cur = db_conn.cursor()
            cur.execute(query, (email, pw))
            if len(cur.fetchall()) != 0:
                self.show_child_data_page()
            else:
                self.login_page.warning.setVisible(False)
                self.login_page.wrong_data.setVisible(True)

    def show_signup_page(self):
        self.signup_page = SignUpPage(self)
        self.setWindowTitle('Sign Up')
        self.setCentralWidget(self.signup_page)
        self.signup_page.submit.clicked.connect(self.submit_user_data_db)

    def submit_user_data_db(self):
        name = self.signup_page.text_edit_fullname.text()
        email = self.signup_page.text_edit_email.text().lower()
        pw = self.signup_page.text_edit_password.text()
        try:
            age = int(self.signup_page.text_edit_age.text())
        except Exception as e:
            age = 0

        gender = self.signup_page.gender
        agree = self.signup_page.agree

        if name == '' or email == '' or pw == '' or age == 0 and gender is None or not agree:
            self.signup_page.warning.setVisible(True)
        else:
            query = ''' SELECT * FROM user_details WHERE Email = ? AND Password = ? '''
            cur = db_conn.cursor()
            cur.execute(query, (email, pw))
            if len(cur.fetchall()) == 0:
                query = ''' INSERT INTO user_details(Name, Password, Email, Gender, Age)
                VALUES(?,?,?,?,?) '''
                cur = db_conn.cursor()
                cur.execute(query, (name, pw, email, gender, age))
                db_conn.commit()
                self.show_login_page()

    def show_msg_box(self):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText('You have to look to the images for at least 3 seconds')
        msg.setWindowTitle('Info')
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok |
                               QtWidgets.QMessageBox.Cancel)
        if msg.exec_() == QtWidgets.QMessageBox.Ok:
            start_test()

    def show_child_data_page(self):
        self.child_data_page = ChildDataPage(self)
        self.setWindowTitle('Child Data')
        self.setCentralWidget(self.child_data_page)
        self.child_data_page.submit.clicked.connect(self.submit_child_data_db)
        self.child_data_page.start_test.clicked.connect(self.show_msg_box)
        self.child_data_page.cvt_sp_hm.clicked.connect(gazeheatplot)
        self.child_data_page.show_result.clicked.connect(
            self.show_prediction_result)
        self.child_data_page.video_image.clicked.connect(gaze_tracking_frame)

    def show_prediction_result(self):
        res = predict_autism()
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(res)
        msg.setWindowTitle('Info')
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok |
                               QtWidgets.QMessageBox.Cancel)
        msg.exec_()

    def submit_child_data_db(self):
        name = self.child_data_page.text_edit_name.text()
        try:
            age = int(self.child_data_page.text_edit_age.text())
        except Exception as e:
            age = 0

        gender = self.child_data_page.gender
        # print('gender =', gender)

        if name == '' or age == 0 and gender is None:
            self.child_data_page.warning.setVisible(True)
            self.child_data_page.success.setVisible(False)
            self.child_data_page.wrong_age.setVisible(False)
        elif 5 < age:
            self.child_data_page.wrong_age.setVisible(True)
            self.child_data_page.warning.setVisible(False)
            self.child_data_page.success.setVisible(False)
        else:
            query = ''' INSERT INTO child_data(Name, Age, Gender)
              VALUES(?,?,?) '''
            cur = db_conn.cursor()
            cur.execute(query, (name, age, gender))
            db_conn.commit()
            self.child_data_page.success.setVisible(True)
            self.child_data_page.wrong_age.setVisible(False)
            self.child_data_page.warning.setVisible(False)
            self.child_data_page.submit.setEnabled(False)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    # win = LoginPage()
    # win = SignUpPage()
    # win = WelcomePage()
    # win = ChildDataPage()
    sys.exit(app.exec_())
