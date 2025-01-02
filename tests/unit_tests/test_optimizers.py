import os
from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH
from si.io.data_file import read_data_file
from si.neural_networks.optimizers import Adam, SGD


class TestOptimizers(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")
        self.w = np.random.rand(self.dataset.X.shape[1], 1)
        self.grad_loss_w = np.random.rand(self.dataset.X.shape[1], 1)

    def testSGD(self):

        sgd = SGD()
        new_w = sgd.update(self.w, self.grad_loss_w)

        self.assertEqual(new_w.shape, self.w.shape)
        self.assertIsNotNone(new_w)
        self.assertTrue(np.all(new_w != self.w))


    def testAdam(self):
        """
Inicialização do Otimizador: 
• Um objeto Adam é criado com uma taxa de aprendizado de 0.001. 
Atualização de Pesos: 
• O método update é chamado com os pesos atuais self.w e o gradiente da perda self.grad_loss_w. 
• O resultado é armazenado em new_w. 
Asserções: 
• assertEqual(new_w.shape, self.w.shape): 
    - Garante que a forma (shape) dos pesos atualizados é igual à dos pesos originais. 
    - Isso é crucial para a compatibilidade dos pesos no modelo. 
• assertIsNotNone(new_w): 
    - Verifica que o método update retorna um valor e que ele não é None. 
• assertTrue(np.all(new_w != self.w)): 
    - Certifica que os novos pesos são diferentes dos originais, o que indica que o algoritmo Adam está 
aplicando as atualizações corretamente. 

Importância do Teste 
1. Validação de Propriedades Fundamentais: 
- Este teste confirma que o otimizador Adam produz uma saída válida e consistente com o formato 
esperado. 
- É essencial garantir que o shape dos pesos não seja alterado durante a atualização. 
2. Confirmação de Atualizações de Pesos: 
- O teste verifica que os pesos foram ajustados em resposta aos gradientes, indicando que o algoritmo 
está funcionando. 
3. Base para Modelos Mais Complexos: 
- O sucesso deste teste é um pré-requisito para o funcionamento de modelos treinados com o Adam, já 
que ele garante que o algoritmo pode manipular os gradientes e atualizar os pesos de forma confiável.

        """
        adam = Adam(learning_rate=0.001)
        new_w = adam.update(self.w, self.grad_loss_w)

        self.assertEqual(new_w.shape, self.w.shape)
        self.assertIsNotNone(new_w)
        self.assertTrue(np.all(new_w != self.w))