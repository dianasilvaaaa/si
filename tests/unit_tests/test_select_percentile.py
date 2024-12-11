from unittest import TestCase

import numpy as np
from datasets import DATASETS_PATH

import os
from si.feature_selection.select_percentile import SelectPercentile
from si.io.csv_file import read_csv

from si.statistics.f_classification import f_classification

class TestSelectPercentile(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv') #carrega o conjunto de dados

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
        #O dataset é lido usando a função read_csv, que retorna um objeto Dataset com as características (features) e os rótulos (label)
    
    def test_fit(self):
        select_percentile = SelectPercentile(score_func = f_classification, percentile= 50)
        #Cria uma instância de SelectPercentile configurada para selecionar as 50% características mais relevantes

        select_percentile.fit(self.dataset)
        self.assertTrue(select_percentile.F.shape[0] > 0)
        self.assertTrue(select_percentile.p.shape[0] > 0)
        #Verifica se os valores F (scores) e p (valores-p) foram corretamente calculados
        #F e p devem ser arrays não vazios

    def test_transform(self):
        #Testa se o seletor funciona ao selecionar 50% das características
        select_percentile = SelectPercentile(score_func = f_classification, percentile= 50)
        select_percentile.fit(self.dataset) ##Ajusta o seletor ao dataset com fit
        new_dataset = select_percentile.transform(self.dataset) ##Transforma o dataset original para obter um novo dataset com todas as características

        self.assertLess(len(new_dataset.features), len(self.dataset.features))
        self.assertLess(new_dataset.X.shape[1], self.dataset.X.shape[1])
        self.assertEqual(len(new_dataset.features),2,"For selecting the top 50% since the dataset has 4 features, the new dataset only should have 2 features")

        #Testa se o seletor funciona ao selecionar 100% das características
        select_percentile = SelectPercentile(score_func = f_classification, percentile= 100)
        select_percentile.fit(self.dataset) #Ajusta o seletor ao dataset com fit
        new_dataset = select_percentile.transform(self.dataset) #Transforma o dataset original para obter um novo dataset com todas as características

        self.assertEqual(len(new_dataset.features), len(self.dataset.features))
        self.assertEqual(new_dataset.X.shape[1], self.dataset.X.shape[1])

        #Testa se o seletor funciona ao selecionar 0% das características
        select_percentile = SelectPercentile(score_func = f_classification, percentile= 0)
        select_percentile.fit(self.dataset)
        new_dataset = select_percentile.transform(self.dataset)

        self.assertEqual(new_dataset.X.shape[1],0)