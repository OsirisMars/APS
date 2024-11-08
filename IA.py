#caso precise faça o:
#pip install torch tensorflow transformers datasets evaluate transformers[torch]

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate

# Exemplo de dataset
frase = [
  "Vixe, essa foi por pouco!",
  "Oxente, eu não esperava por isso!",
  "Vixe Maria, que confusão!",
  "Oxente, que surpresa boa!",
  "Vixe do céu, olha só quem chegou!",
  "Oxente, parece que vai chover!",
  "Vixe, como você cresceu!",
  "Oxente, isso não me parece certo.",
  "Vixe, nunca pensei que veria isso.",
  "Oxente, que calor é esse?",

  # Frases sem "Vixe" ou "Oxente" (rótulo 0)
  "Hoje o dia está muito tranquilo.",
  "Será que vai chover amanhã?",
  "Aquela festa foi incrível!",
  "Acho que vou precisar de mais café.",
  "Gostei do novo restaurante na cidade.",
  "Ele não sabia o que dizer.",
  "A manhã está ensolarada e calma.",
  "Vamos tentar fazer isso mais tarde.",
  "Eu prefiro ficar em casa hoje.",
  "Parece que teremos uma semana movimentada."
]
label = [1 if "vixe" in frase.lower() or "oxente" in frase.lower() else 0 for frase in frase]


dataset_df = pd.DataFrame({"text": frase, "label": label})

dataset_df.to_csv("dataset_vixe_oxente.csv", index=False)
dataset = Dataset.from_pandas(dataset_df)
print("Dataset ta feito meu mano")




# Baixando o tokenizador e o modelo
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Tokenizando as frases
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Dividindo o dataset em treino e validação
dataset = tokenized_datasets.train_test_split(test_size=0.1)

# Configurando o treinamento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# Treinamento do modelo
trainer.train()

acuracia = evaluate.load("precision")
results = acuracia.compute(references=[0, 1], predictions=[0, 1])
print(results)

