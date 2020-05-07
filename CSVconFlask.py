# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:16:41 2020

@author: steza
"""
import csv
from flask import Flask
from flask import jsonify


app = Flask(__name__)
filename = 'C:/Users/steza/Documentos/iris.csv'
raw_data = open(filename, 'r')
reader = csv.reader( raw_data , delimiter =';', quoting = csv.QUOTE_NONE)
lista = list(reader)

#varia = [1,3]
@app.route('/')
def archivo():
    return jsonify(lista)
        

if __name__ == '__main__':
    app.run()
