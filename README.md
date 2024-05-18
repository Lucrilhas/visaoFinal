
### Link dos pesos:

https://drive.google.com/drive/folders/1MxuBeDI1OrYix-IKOWJdYbgT_GDp0v8K?usp=sharing

### Para gerar os outros Datasets:
Descomentar a célula no notebook "preprocess.ipynb"

### Arquivo "run_model.py"
* Roda o Moblie Net v2 de acordo com a especificação do argumento passado.
* Também salva os pesos e os dados obtidos em diferentes csv.
* Esses dados são:
    * hist: Historico do treino e validação.
    * res: Resultado do treino, val e teste e outras especificações do modelo.
    * cm: Confusion matrix dos testes.

### Notebook "comandos.ipynb":
* Executa as variações do modelo. 
    * OBS: Fiz desse jeito pois assim não acontecia estouro de memória ao treinar o modelo várias vezes seguidas no mesmo notebook.
* Lê os resultados e gera gráficos correspondentes

### Notebook "preprocess.ipynb":
* Etapas do pré-processaamento realizadas.

