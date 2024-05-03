import cv2
import os
from flask import Flask, request, render_template, redirect, url_for, session, flash
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import secrets
import requests
import hashlib

# Defining Flask App
app = Flask(__name__)
app.secret_key = secrets.token_hex(24)  # Set a secure random key for session management

# Define a dictionary to store user credentials (temporary storage for demonstration)
user_credentials = {}

# Replace these with your reCAPTCHA keys
RECAPTCHA_SITE_KEY = '6LfxQMkpAAAAAJgWY4LmbE2Pp51sG_ryc9nKwJVP'
RECAPTCHA_SECRET_KEY = '6Lc_OMkpAAAAAGgF1zjRVbDgEKctlZN_mIgsNsYF'

nimgs = 10

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initializing VideoCapture object to access Webcam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

# extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

# Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

## A function to get names and roll numbers of all users
def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l

## A function to delete a user folder
def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(os.path.join(duser, i))
    os.rmdir(duser)

################## ROUTING FUNCTIONS #########################

# Home page
@app.route('/')
def home():
    return render_template('home.html')

# Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == 'admin' and password == 'password':
            session['logged_in'] = True
            return redirect(url_for('admin_page'))  # Redirect to admin_page
        if username == 'user' and password == 'user':
            session['username'] = True
            return redirect(url_for('user_page'))  # Redirect to admin_page

        # Check if the username exists in user_credentials or new_users.csv
        if username in user_credentials:
            stored_password = user_credentials[username]['password']
        else:
            # Fetch the stored password from new_users.csv
            user_data = pd.read_csv('new_users.csv', names=['username', 'userid', 'password', 'phone'])
            user_row = user_data[user_data['username'] == username]
            if user_row.empty:
                error_message = 'Invalid credentials. Please try again.'
                return render_template('login.html', error=error_message)
            stored_password = user_row.iloc[0]['password']

        # Hash the provided password and compare with stored hashed password
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        if hashed_password == stored_password:
            # Successful login, set session and redirect
            session['username'] = username
            return redirect(url_for('user_page'))

        error_message = 'Invalid credentials. Please try again.'
        return render_template('login.html', error=error_message)

    return render_template('login.html')

# User page
@app.route('/user')
def user_page():
    if 'username' not in session:
        return redirect(url_for('login'))
    names, rolls, times, l = extract_attendance()
    userlist, _, _, _ = getallusers()  # Extract userlist from getallusers function
    # Retrieve user-specific data or perform actions based on session username
    return render_template('user.html', username=session['username'], names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, userlist=userlist)
    # return render_template('user.html', username=session['username'])

# Admin Page
@app.route('/admin')
def admin_page():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('login'))
    
    names, rolls, times, l = extract_attendance()
    userlist, _, _, _ = getallusers()  # Extract userlist from getallusers function
    return render_template('admin.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, userlist=userlist)

## List users page
@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)

## Delete functionality
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder('static/faces/'+duser)

    # Remove user's entry from new_users.csv
    user_data = pd.read_csv('new_users.csv', names=['username', 'userid', 'password', 'phone'])
    user_data = user_data[~user_data['username'].str.startswith(duser.split('_')[0])]  # Remove entries for this user
    user_data.to_csv('new_users.csv', index=False)  # Save updated CSV file

    ## if all the face are deleted, delete the trained file...
    if not os.listdir('static/faces/'):
        os.remove('static/face_recognition_model.pkl')

    try:
        train_model()
        flash('User deleted successfully!', 'success')  # Flash success message
    except Exception as e:
        flash('Error deleting user.', 'error')  # Flash error message

    userlist, names, rolls, l = getallusers()
    # Pass all required variables to the template
    return render_template('admin.html', userlist=userlist, names=names, rolls=rolls, l=l, times=[], totalreg=totalreg(), datetoday2=datetoday2)


# Our main Face Recognition functionality
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# A function to add a new user.
# This function will run when we add a new user.
@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        newpassword = request.form['newpassword']
        newphone = request.form['newphone']

        # Check if username already exists
        if newusername in user_credentials:
            error_message = 'Username already exists. Please choose a different username.'
            flash(error_message, 'error')  # Flash error message
            return redirect(url_for('admin_page'))
    
        # Check if username already exists in the CSV file
        user_data = pd.read_csv('new_users.csv', names=['username', 'userid', 'password', 'phone'])
        if newusername in user_data['username'].values:
            error_message = 'Username already exists. Please choose a different username.'
            flash(error_message, 'error')  # Flash error message
            return redirect(url_for('admin_page'))

        # Hash the password using SHA-256 for secure storage
        hashed_password = hashlib.sha256(newpassword.encode()).hexdigest()

        # Store user credentials (username and hashed password) in dictionary
        user_credentials[newusername] = {'password': hashed_password, 'phone': newphone}

        # Save new user details to a CSV file or database
        with open('new_users.csv', 'a') as f:
            f.write(f'{newusername},{newuserid},{newpassword},{newphone}\n')

        userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        i, j = 0, 0
        cap = cv2.VideoCapture(0)
        while 1:
            _, frame = cap.read()
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                if j % 5 == 0:
                    name = newusername+'_'+str(i)+'.jpg'
                    cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                    i += 1
                j += 1
            if j == nimgs*5:
                break
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        print('Training Model')
        train_model()
        names, rolls, times, l = extract_attendance()
        flash('User registered successfully!', 'success')
        return redirect(url_for('admin_page'))

    return redirect(url_for('admin_page'))
# Logout
@app.route('/logout', methods=['POST'])
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('login'))

# Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)