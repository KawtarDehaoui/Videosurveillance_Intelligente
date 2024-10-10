from flask import Blueprint, render_template, request, flash, redirect, url_for,Response
from .models import User
from werkzeug.security import generate_password_hash, check_password_hash
from . import db
from flask_login import login_user, login_required, logout_user, current_user

from .place_count import places
from .accident_detec import accident
from .ANPR import matr
from .ANPR import table_data



auth = Blueprint('auth', __name__)



@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password1')

        user = User.query.filter_by(email=email).first()
        if user:
            if check_password_hash(user.password, password):
                flash('Logged in successfully!', category='success')
                login_user(user, remember=True)
                return redirect(url_for('views.home'))
            else:
                flash('Incorrect password, try again.', category='error')
        else:
            flash('Email does not exist.', category='error')

    return render_template("login.html", user=current_user)


@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))





@auth.route('/About_us')
def About_us():
    return render_template("About_us.html", user=current_user)

@auth.route('/users',methods=['GET', 'POST'])
@login_required
def Users():
    if request.method == 'POST':
        email = request.form.get('email')
        first_name = request.form.get('firstName')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')

        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already exists.', category='error')
        elif len(email) < 4:
            flash('Email must be greater than 3 characters.', category='error')
        elif len(first_name) < 2:
            flash('First name must be greater than 1 character.', category='error')
        elif password1 != password2:
            flash('Passwords don\'t match.', category='error')
        elif len(password1) < 7:
            flash('Password must be at least 7 characters.', category='error')
        else:
            new_user = User(email=email, first_name=first_name, password=generate_password_hash(
                password1, method='sha256'))
            db.session.add(new_user)
            db.session.commit()
            #login_user(new_user, remember=True)
            flash('Account created!', category='success')
            return redirect(url_for('auth.Users'))
    elif request.method == 'GET':
        users = User.query.all()
        return render_template('Users.html', us=users, user=current_user)

@auth.route('/delete_user/<int:user_id>', methods=['GET'])
@login_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    flash('User deleted successfully!', category='success')
    return redirect(url_for('auth.Users'))


@auth.route('/count')
@login_required
def count_places():
    return render_template('videos.html',user=current_user)

@auth.route('/video')
def video_place():
    return Response(places(),mimetype='multipart/x-mixed-replace; boundary=frame')

@auth.route('/accident')
def video_accident():
    return Response(accident(),mimetype='multipart/x-mixed-replace; boundary=frame')

@auth.route('/anpr')
@login_required
def anpr():
    return Response(matr(),mimetype='multipart/x-mixed-replace; boundary=frame')

@auth.route('/anprht')
@login_required
def anprht():
    return render_template('anpr.html', data=table_data[::-1], user=current_user)  # Pass table data to the HTML template




