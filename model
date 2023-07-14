pip install transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import torch

# Load pre-trained T5 model and tokenizer
model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load and preprocess your labeled dataset
dataset = [...]  # Load or create your labeled dataset

# Tokenize and process the dataset
inputs = tokenizer.prepare_seq2seq_batch(dataset, padding=True, truncation=True, return_tensors='pt')

# Fine-tune the model on your dataset
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set the model to training mode
model.train()

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    optimizer.zero_grad()
    inputs = inputs.to(device)
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

# Save the fine-tuned model
model.save_pretrained('fine_tuned_model')
# Load the fine-tuned model
model = T5ForConditionalGeneration.from_pretrained('fine_tuned_model')
tokenizer = T5Tokenizer.from_pretrained('fine_tuned_model')

# Prepare input text for inference
input_text = '...'  # Provide the input text for summarization
inputs = tokenizer.prepare_seq2seq_batch([input_text], padding=True, truncation=True, return_tensors='pt')

# Set the model to evaluation mode
model.eval()

# Generate the summary
inputs = inputs.to(device)
generated_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Print the generated summary
print('Generated Summary:', summary)
