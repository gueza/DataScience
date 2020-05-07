# -*- coding: utf-8 -*-
"""
Created on Mon May  4 22:06:04 2020

@author: steza
"""

from datetime import datetime, date, time, timedelta
from flask import Flask
from flask import jsonify

app = Flask(__name__)

ahora = datetime.now() 
datetime= {'Hora' : ahora.hour, 'Minutos': ahora.minute, 'Segundos': ahora.second, 'Fecha': [ahora.year, ahora.month, ahora.day]}


@app.route('/')
def get_current_user():
    return jsonify(datetime)

if __name__ == '__main__':
    app.run()