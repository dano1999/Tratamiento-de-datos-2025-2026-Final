from pathlib import Path

def run():
    
    import numpy as np
    import pandas as pd
    import re

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.utils.class_weight import compute_class_weight

    from gensim.models import Word2Vec

    from sentence_transformers import SentenceTransformer

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader, Dataset

    from collections import Counter
    from collections import defaultdict


    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.optim import AdamW

    import matplotlib
    import matplotlib.pyplot as plt


    train_path = "data/datasets/rumoureval2019_train.csv"
    val_path   = "data/datasets/rumoureval2019_val.csv"
    test_path  = "data/datasets/rumoureval2019_test.csv"

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)

    print("NaN en label (antes de limpiar):")
    print("  train:", train_df["label"].isna().sum())
    print("  val:  ", val_df["label"].isna().sum())
    print("  test: ", test_df["label"].isna().sum())

    train_df = train_df.dropna(subset=["label"])
    val_df   = val_df.dropna(subset=["label"])
    test_df  = test_df.dropna(subset=["label"])

    print("\nNaN en label (después de limpiar):")
    print("  train:", train_df["label"].isna().sum())
    print("  val:  ", val_df["label"].isna().sum())
    print("  test: ", test_df["label"].isna().sum())

    print("\nEtiquetas únicas en train:", train_df["label"].unique())

    print("\nDistribución de clases:")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = df["label"].value_counts()
        total = counts.sum()
        print(f"\n {name.upper()} (total = {total}) ")
        for label, c in counts.items():
            print(f"{label:8s}: {c:4d} ({c/total:.3f})")

    def concat_text_row(row):
        src = row.get("source_text", "")
        rep = row.get("reply_text", "")
        src = "" if pd.isna(src) else str(src)
        rep = "" if pd.isna(rep) else str(rep)
        return (src + " [SEP] " + rep).strip()

    X_train_text = train_df.apply(concat_text_row, axis=1).tolist()
    y_train = train_df["label"].values         

    X_val_text = val_df.apply(concat_text_row, axis=1).tolist()
    y_val = val_df["label"].values

    X_test_text = test_df.apply(concat_text_row, axis=1).tolist()
    y_test = test_df["label"].values

    print("\nEjemplo de texto de entrenamiento:")
    print(X_train_text[0])
    print("Etiqueta:", y_train[0])

    label_encoder = LabelEncoder()
    y_train_idx = label_encoder.fit_transform(y_train)
    y_val_idx   = label_encoder.transform(y_val)
    y_test_idx  = label_encoder.transform(y_test)
    num_classes = len(label_encoder.classes_)
    print("\nClases (label_encoder):", label_encoder.classes_)

    major_class = Counter(y_test).most_common(1)[0][0]
    baseline_acc = np.mean(y_test == major_class)
    print(f"\nClase mayoritaria en TEST: {major_class}")
    print(f"Accuracy baseline (siempre '{major_class}') = {baseline_acc:.4f}")


    def train_and_evaluate_knn(X_train_vec, y_train,
                               X_val_vec, y_val,
                               X_test_vec, y_test,
                               k_values=[1, 3, 5, 7, 9],
                               title=""):
        print("RESULTADOS KNN -", title)

        best_k = None
        best_acc = 0.0

        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_vec, y_train)
            y_val_pred = knn.predict(X_val_vec)
            acc_val = accuracy_score(y_val, y_val_pred)
            print(f"k = {k} --> Accuracy validación = {acc_val:.4f}")

            if acc_val > best_acc:
                best_acc = acc_val
                best_k = k

        print("\nMejor número de vecinos (k) encontrado en validación:", best_k)
        print(f"Accuracy de validación con k={best_k}: {best_acc:.4f}")

        final_knn = KNeighborsClassifier(n_neighbors=best_k)
        final_knn.fit(X_train_vec, y_train)

        y_test_pred = final_knn.predict(X_test_vec)
        acc_test = accuracy_score(y_test, y_test_pred)

        print(f"\nAccuracy en TEST con k={best_k}: {acc_test:.4f}")
        print("\nClassification report (TEST):")
        print(classification_report(y_test, y_test_pred, digits=4))

        print("\nEjemplo de predicciones en test (primeros 20):")
        print("y_test_pred[:20] =", y_test_pred[:20])
        print("y_test[:20]      =", y_test[:20])

        return final_knn, best_k, acc_test


    class ConvNet1D(nn.Module):

        def __init__(self, input_dim, num_classes, dropout=0.3):
            super(ConvNet1D, self).__init__()

            self.conv1 = nn.Conv1d(in_channels=1,   out_channels=64, kernel_size=5, padding=2)
            self.bn1   = nn.BatchNorm1d(64)

            self.conv2 = nn.Conv1d(in_channels=64,  out_channels=64, kernel_size=5, padding=2)
            self.bn2   = nn.BatchNorm1d(64)

            self.conv3 = nn.Conv1d(in_channels=64,  out_channels=64, kernel_size=5, padding=2)
            self.bn3   = nn.BatchNorm1d(64)

            self.conv4 = nn.Conv1d(in_channels=64,  out_channels=64, kernel_size=5, padding=2)
            self.bn4   = nn.BatchNorm1d(64)

            self.global_pool = nn.AdaptiveMaxPool1d(1)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(64, num_classes)

        def forward(self, x):
            x = x.unsqueeze(1)           # (B, 1, L)

            x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, L)
            x = F.relu(self.bn2(self.conv2(x)))  # (B, 64, L)
            x = F.relu(self.bn3(self.conv3(x)))  # (B, 64, L)
            x = F.relu(self.bn4(self.conv4(x)))  # (B, 64, L)

            x = self.global_pool(x)      # (B, 64, 1)
            x = x.squeeze(-1)            # (B, 64)
            x = self.dropout(x)
            x = self.fc(x)               # (B, num_classes)
            return x



    def train_and_evaluate_cnn(
        X_train, y_train_idx,
        X_val, y_val_idx,
        X_test, y_test_idx,
        label_encoder,
        title="CNN",
        num_epochs=20,
        batch_size=32,
        lr=5e-4,
        dropout=0.3,
        device=None
    ):

        print("ENTRENANDO RED NEURONAL CONVOLUCIONAL -", title)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Usando dispositivo:", device)

        X_train = np.asarray(X_train, dtype=np.float32)
        X_val   = np.asarray(X_val,   dtype=np.float32)
        X_test  = np.asarray(X_test,  dtype=np.float32)

        y_train_idx = np.asarray(y_train_idx, dtype=np.int64)
        y_val_idx   = np.asarray(y_val_idx,   dtype=np.int64)
        y_test_idx  = np.asarray(y_test_idx,  dtype=np.int64)

        input_dim = X_train.shape[1]
        num_classes = len(label_encoder.classes_)

        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_idx))
        val_dataset   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val_idx))
        test_dataset  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test_idx))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        class_weights_np = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(num_classes),
            y=y_train_idx
        )
        class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
        print("\nPesos de clase (para CrossEntropyLoss):")
        for idx, w in enumerate(class_weights_np):
            print(f"  Clase {idx} ({label_encoder.classes_[idx]}): {w:.4f}")

        model = ConvNet1D(input_dim=input_dim, num_classes=num_classes, dropout=dropout).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        best_val_acc = 0.0
        best_state_dict = None

        for epoch in range(1, num_epochs + 1):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_X.size(0)
                _, preds = torch.max(outputs, 1)
                correct_train += (preds == batch_y).sum().item()
                total_train += batch_X.size(0)

            train_loss = running_loss / total_train
            train_acc = correct_train / total_train

            model.eval()
            correct_val = 0
            total_val = 0
            with torch.inference_mode():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    outputs = model(batch_X)
                    _, preds = torch.max(outputs, 1)
                    correct_val += (preds == batch_y).sum().item()
                    total_val += batch_X.size(0)

            val_acc = correct_val / total_val

            print(f"Época {epoch:02d}/{num_epochs} | "
                  f"Loss train = {train_loss:.4f} | "
                  f"Acc train = {train_acc:.4f} | "
                  f"Acc val = {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state_dict = model.state_dict()

        print(f"\nMejor accuracy de validación alcanzado: {best_val_acc:.4f}")

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        model.eval()
        all_preds = []
        all_true = []
        with torch.inference_mode():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                _, preds = torch.max(outputs, 1)
                all_preds.append(preds.cpu().numpy())
                all_true.append(batch_y.numpy())

        all_preds = np.concatenate(all_preds)
        all_true  = np.concatenate(all_true)

        y_test_pred_labels = label_encoder.inverse_transform(all_preds)
        y_test_true_labels = label_encoder.inverse_transform(all_true)

        acc_test = accuracy_score(y_test_true_labels, y_test_pred_labels)
        print(f"\nAccuracy en TEST ({title}) = {acc_test:.4f}")
        print("\nClassification report (TEST):")
        print(classification_report(y_test_true_labels, y_test_pred_labels, digits=4))

        print("\nEjemplo de predicciones en test (primeros 20):")
        print("y_test_pred[:20] =", y_test_pred_labels[:20])
        print("y_test[:20]      =", y_test_true_labels[:20])

        return model, acc_test, y_test_pred_labels



    print("\n\nTF-IDF + KNN")

    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
    X_val_tfidf   = tfidf_vectorizer.transform(X_val_text)
    X_test_tfidf  = tfidf_vectorizer.transform(X_test_text)

    knn_tfidf, best_k_tfidf, acc_test_tfidf = train_and_evaluate_knn(
        X_train_tfidf, y_train,
        X_val_tfidf, y_val,
        X_test_tfidf, y_test,
        k_values=[1, 3, 5, 7, 9],
        title="TF-IDF"
    )

    y_pred_test = knn_tfidf.predict(X_test_tfidf)
    print("\nPREDICCIÓN TF-IDF + KNN (primeras 10 líneas)")
    for i in range(10):
        print(f"Texto {i}:")
        print("   Predicción:", y_pred_test[i])
        print("   Real:      ", y_test[i])


    print("\n\nTF-IDF + CNN")

    scaler_tfidf = StandardScaler(with_mean=False)
    X_train_tfidf_scaled = scaler_tfidf.fit_transform(X_train_tfidf)
    X_val_tfidf_scaled   = scaler_tfidf.transform(X_val_tfidf)
    X_test_tfidf_scaled  = scaler_tfidf.transform(X_test_tfidf)

    X_train_tfidf_dense = X_train_tfidf_scaled.toarray()
    X_val_tfidf_dense   = X_val_tfidf_scaled.toarray()
    X_test_tfidf_dense  = X_test_tfidf_scaled.toarray()

    cnn_tfidf, acc_test_tfidf_cnn, y_pred_test_tfidf_cnn = train_and_evaluate_cnn(
        X_train_tfidf_dense, y_train_idx,
        X_val_tfidf_dense,   y_val_idx,
        X_test_tfidf_dense,  y_test_idx,
        label_encoder,
        title="TF-IDF + CNN",
        num_epochs=15,       
        batch_size=32,
        lr=5e-4,
        dropout=0.3
    )

    print("\nPREDICCIÓN TF-IDF + CNN (primeras 10 líneas)")
    for i in range(10):
        print(f"Texto {i}:")
        print("   Predicción:", y_pred_test_tfidf_cnn[i])
        print("   Real:      ", y_test[i])


    print("\n\nWord2Vec + KNN")

    def simple_tokenize(text):
        return str(text).lower().split()

    train_tokens = [simple_tokenize(t) for t in X_train_text]
    val_tokens   = [simple_tokenize(t) for t in X_val_text]
    test_tokens  = [simple_tokenize(t) for t in X_test_text]

    w2v_model = Word2Vec(
        sentences=train_tokens,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        sg=1
    )

    word_vectors = w2v_model.wv

    def document_embedding(tokens, word_vectors, dim=100):
        vecs = []
        for tok in tokens:
            if tok in word_vectors:
                vecs.append(word_vectors[tok])
        if len(vecs) == 0:
            return np.zeros(dim)
        else:
            return np.mean(vecs, axis=0)

    def build_doc_matrix(list_of_tokens, word_vectors, dim=100):
        return np.vstack([
            document_embedding(toks, word_vectors, dim)
            for toks in list_of_tokens
        ])

    X_train_w2v = build_doc_matrix(train_tokens, word_vectors, dim=100)
    X_val_w2v   = build_doc_matrix(val_tokens,   word_vectors, dim=100)
    X_test_w2v  = build_doc_matrix(test_tokens,  word_vectors, dim=100)

    knn_w2v, best_k_w2v, acc_test_w2v = train_and_evaluate_knn(
        X_train_w2v, y_train,
        X_val_w2v, y_val,
        X_test_w2v, y_test,
        k_values=[1, 3, 5, 7, 9],
        title="Word2Vec (media embeddings)"
    )

    y_pred_test_w2v = knn_w2v.predict(X_test_w2v)
    print("\nPREDICCIÓN Word2Vec + KNN (primeras 10 líneas)")
    for i in range(10):
        print(f"{i}) pred={y_pred_test_w2v[i]}  real={y_test[i]}")


    print("\n\nWord2Vec + CNN")

    scaler_w2v = StandardScaler()
    X_train_w2v_scaled = scaler_w2v.fit_transform(X_train_w2v)
    X_val_w2v_scaled   = scaler_w2v.transform(X_val_w2v)
    X_test_w2v_scaled  = scaler_w2v.transform(X_test_w2v)

    cnn_w2v, acc_test_w2v_cnn, y_pred_test_w2v_cnn = train_and_evaluate_cnn(
        X_train_w2v_scaled, y_train_idx,
        X_val_w2v_scaled,   y_val_idx,
        X_test_w2v_scaled,  y_test_idx,
        label_encoder,
        title="Word2Vec + CNN (mejorada)",
        num_epochs=35,       
        batch_size=32,
        lr=3e-4,             
        dropout=0.4
    )

    print("\nPREDICCIÓN Word2Vec + CNN (primeras 10 líneas)")
    for i in range(10):
        print(f"{i}) pred={y_pred_test_w2v_cnn[i]}  real={y_test[i]}")


    print("\n\nEMBEDDINGS (Sentence-BERT) + KNN")

    bert_model_st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    X_train_bert = bert_model_st.encode(X_train_text, batch_size=32, show_progress_bar=True)
    X_val_bert   = bert_model_st.encode(X_val_text,   batch_size=32, show_progress_bar=True)
    X_test_bert  = bert_model_st.encode(X_test_text,  batch_size=32, show_progress_bar=True)

    knn_bert, best_k_bert, acc_test_bert = train_and_evaluate_knn(
        X_train_bert, y_train,
        X_val_bert,   y_val,
        X_test_bert,  y_test,
        k_values=[1, 3, 5, 7, 9],
        title="Embeddings contextuales (Sentence-BERT)"
    )

    y_pred_test_bert = knn_bert.predict(X_test_bert)
    print("\nPREDICCIÓN BERT + KNN (primeras 10 líneas)")
    for i in range(10):
        print(f"{i}) pred={y_pred_test_bert[i]}  real={y_test[i]}")

    print("\n\nBERT Embeddings + CNN ")

    scaler_bert = StandardScaler()
    X_train_bert_scaled = scaler_bert.fit_transform(X_train_bert)
    X_val_bert_scaled   = scaler_bert.transform(X_val_bert)
    X_test_bert_scaled  = scaler_bert.transform(X_test_bert)

    cnn_bert, acc_test_bert_cnn, y_pred_test_bert_cnn = train_and_evaluate_cnn(
        X_train_bert_scaled, y_train_idx,
        X_val_bert_scaled,   y_val_idx,
        X_test_bert_scaled,  y_test_idx,
        label_encoder,
        title="Sentence-BERT + CNN",
        num_epochs=30,       
        batch_size=32,
        lr=5e-4,
        dropout=0.3
    )

    print("\nPREDICCIÓN BERT + CNN (primeras 10 líneas)")
    for i in range(10):
        print(f"{i}) pred={y_pred_test_bert_cnn[i]}  real={y_test[i]}")


    print("\n\nTRANSFORMER PREENTRENADO + FINE-TUNING")

    transformer_model_name = "distilbert-base-uncased"
    tokenizer_hf = AutoTokenizer.from_pretrained(transformer_model_name)

    def tokenize_batch_hf(texts, tokenizer, max_length=128):
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

    train_encodings_hf = tokenize_batch_hf(X_train_text, tokenizer_hf)
    val_encodings_hf   = tokenize_batch_hf(X_val_text,   tokenizer_hf)
    test_encodings_hf  = tokenize_batch_hf(X_test_text,  tokenizer_hf)

    class RumourEvalHFDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    train_dataset_hf = RumourEvalHFDataset(train_encodings_hf, y_train_idx)
    val_dataset_hf   = RumourEvalHFDataset(val_encodings_hf,   y_val_idx)
    test_dataset_hf  = RumourEvalHFDataset(test_encodings_hf,  y_test_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo para Transformer:", device)

    model_hf = AutoModelForSequenceClassification.from_pretrained(
        transformer_model_name,
        num_labels=num_classes
    ).to(device)

    optimizer_hf = AdamW(model_hf.parameters(), lr=2e-5)

    train_loader_hf = DataLoader(train_dataset_hf, batch_size=16, shuffle=True)
    val_loader_hf   = DataLoader(val_dataset_hf,   batch_size=32, shuffle=False)
    test_loader_hf  = DataLoader(test_dataset_hf,  batch_size=32, shuffle=False)

    num_epochs_hf = 3
    best_val_acc_hf = 0.0
    best_state_dict_hf = None

    for epoch in range(1, num_epochs_hf + 1):
        model_hf.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch in train_loader_hf:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer_hf.zero_grad()
            outputs = model_hf(**batch)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer_hf.step()

            total_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(dim=-1)
            correct_train += (preds == batch["labels"]).sum().item()
            total_train += batch["labels"].size(0)

        train_loss = total_loss / total_train
        train_acc = correct_train / total_train

        model_hf.eval()
        correct_val = 0
        total_val = 0
        with torch.inference_mode():
            for batch in val_loader_hf:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model_hf(**batch)
                logits = outputs.logits
                preds = logits.argmax(dim=-1)
                correct_val += (preds == batch["labels"]).sum().item()
                total_val += batch["labels"].size(0)

        val_acc = correct_val / total_val

        print(f"[Transformer] Época {epoch}/{num_epochs_hf} | "
              f"Loss train = {train_loss:.4f} | Acc train = {train_acc:.4f} | Acc val = {val_acc:.4f}")

        if val_acc > best_val_acc_hf:
            best_val_acc_hf = val_acc
            best_state_dict_hf = model_hf.state_dict()

    print(f"\nMejor accuracy de validación (Transformer) = {best_val_acc_hf:.4f}")

    if best_state_dict_hf is not None:
        model_hf.load_state_dict(best_state_dict_hf)

    model_hf.eval()
    all_preds_hf = []
    all_true_hf = []

    with torch.inference_mode():
        for batch in test_loader_hf:
            labels = batch["labels"].numpy().copy()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model_hf(**batch)
            logits = outputs.logits
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds_hf.append(preds)
            all_true_hf.append(labels)

    all_preds_hf = np.concatenate(all_preds_hf)
    all_true_hf  = np.concatenate(all_true_hf)

    acc_test_transformer = accuracy_score(all_true_hf, all_preds_hf)
    print(f"\nAccuracy en TEST (Transformer fine-tuned: {transformer_model_name}) = {acc_test_transformer:.4f}")

    y_test_pred_labels_transformer = label_encoder.inverse_transform(all_preds_hf)
    y_test_true_labels = label_encoder.inverse_transform(all_true_hf)

    print("\nClassification report (TEST) - Transformer fine-tuned:")
    print(classification_report(y_test_true_labels, y_test_pred_labels_transformer, digits=4))


    print("\n\nRESUMEN FINAL - KNN")
    print(f"TF-IDF (KNN):        mejor k = {best_k_tfidf},  accuracy test = {acc_test_tfidf:.4f}")
    print(f"Word2Vec (KNN):      mejor k = {best_k_w2v},    accuracy test = {acc_test_w2v:.4f}")
    print(f"Sentence-BERT (KNN): mejor k = {best_k_bert},   accuracy test = {acc_test_bert:.4f}")

    print("\nRESUMEN FINAL - CNN")
    print(f"TF-IDF  + CNN:        accuracy test = {acc_test_tfidf_cnn:.4f}")
    print(f"Word2Vec + CNN:       accuracy test = {acc_test_w2v_cnn:.4f}")
    print(f"Sentence-BERT + CNN:  accuracy test = {acc_test_bert_cnn:.4f}")
    print(f"\nBaseline mayoría ('{major_class}') en TEST: accuracy = {baseline_acc:.4f}")

    print("\nRESUMEN FINAL - TRANSFORMER FINE-TUNED")
    print(f"Transformer ({transformer_model_name}): accuracy test = {acc_test_transformer:.4f}")


    print("\n\nBASELINE LÉXICA + URL (MEJORADA, DATA-DRIVEN)")


    _url_re = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

    def extract_urls(text):
        return _url_re.findall(str(text))

    def extract_domain(url):
        u = url.lower()
        u = u.replace("http://", "").replace("https://", "")
        if u.startswith("www."):
            u = u[4:]
        return u.split("/")[0].strip()

    _token_re = re.compile(r"[a-zA-Z']+")

    def tokenize_basic(text):

        return _token_re.findall(str(text).lower())

    def bigrams(tokens):
        return [tokens[i] + "_" + tokens[i+1] for i in range(len(tokens)-1)]



    labels = list(label_encoder.classes_)
    label_set = set(labels)


    tok_counts = {lab: Counter() for lab in labels}
    bi_counts  = {lab: Counter() for lab in labels}
    total_tok  = {lab: 0 for lab in labels}
    total_bi   = {lab: 0 for lab in labels}


    domain_counts = {lab: Counter() for lab in labels}

    for text, lab in zip(X_train_text, y_train):
        tks = tokenize_basic(text)
        tok_counts[lab].update(tks)
        total_tok[lab] += len(tks)

        bis = bigrams(tks)
        bi_counts[lab].update(bis)
        total_bi[lab] += len(bis)

        for u in extract_urls(text):
            dom = extract_domain(u)
            if dom:
                domain_counts[lab][dom] += 1


    vocab_tok = set()
    vocab_bi  = set()
    for lab in labels:
        vocab_tok |= set(tok_counts[lab].keys())
        vocab_bi  |= set(bi_counts[lab].keys())

    V_tok = len(vocab_tok) if len(vocab_tok) > 0 else 1
    V_bi  = len(vocab_bi)  if len(vocab_bi)  > 0 else 1


    alpha_tok = 0.5
    alpha_bi  = 0.5

    def build_log_odds_lexicon(counts_by_class, totals_by_class, V, top_k=120):

        lexicon = {lab: {} for lab in labels}


        total_all = sum(totals_by_class.values())

        for lab in labels:
            total_lab = totals_by_class[lab]
            total_oth = total_all - total_lab


            others = Counter()
            for other_lab in labels:
                if other_lab != lab:
                    others.update(counts_by_class[other_lab])

            scored = []
            for term in counts_by_class[lab].keys():
                c_lab = counts_by_class[lab][term]
                c_oth = others.get(term, 0)

                p_lab = (c_lab + alpha_tok) / (total_lab + alpha_tok * V)
                p_oth = (c_oth + alpha_tok) / (total_oth + alpha_tok * V)

                score = np.log(p_lab) - np.log(p_oth)
                scored.append((term, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            for term, score in scored[:top_k]:
                lexicon[lab][term] = score

        return lexicon

    lex_tok = build_log_odds_lexicon(tok_counts, total_tok, V_tok, top_k=200)
    lex_bi  = build_log_odds_lexicon(bi_counts,  total_bi,  V_bi,  top_k=120)


    min_dom_freq = 3
    domain_global = Counter()
    for lab in labels:
        domain_global.update(domain_counts[lab])

    domains_kept = {d for d, c in domain_global.items() if c >= min_dom_freq}


    dom_scores = {lab: {} for lab in labels}
    alpha_dom = 0.5
    V_dom = len(domains_kept) if len(domains_kept) > 0 else 1

    total_dom = {lab: sum(domain_counts[lab][d] for d in domains_kept) for lab in labels}
    total_dom_all = sum(total_dom.values())

    for lab in labels:

        others = Counter()
        for other_lab in labels:
            if other_lab != lab:
                for d in domains_kept:
                    others[d] += domain_counts[other_lab][d]

        total_lab = total_dom[lab]
        total_oth = total_dom_all - total_lab

        for d in domains_kept:
            c_lab = domain_counts[lab][d]
            c_oth = others[d]

            p_lab = (c_lab + alpha_dom) / (total_lab + alpha_dom * V_dom)
            p_oth = (c_oth + alpha_dom) / (total_oth + alpha_dom * V_dom)

            dom_scores[lab][d] = float(np.log(p_lab) - np.log(p_oth))


    w_tok = 1.0
    w_bi  = 0.8
    w_dom = 1.2
    w_qmark = 0.8  

    def predict_lex_url(text, majority_label):
        t = str(text)
        t_low = t.lower()
        tks = tokenize_basic(t_low)
        bis = bigrams(tks)
        urls = extract_urls(t_low)
        doms = [extract_domain(u) for u in urls]
        doms = [d for d in doms if d]

        scores = {lab: 0.0 for lab in labels}


        for tok in tks:
            for lab in labels:
                if tok in lex_tok[lab]:
                    scores[lab] += w_tok * lex_tok[lab][tok]


        for bi in bis:
            for lab in labels:
                if bi in lex_bi[lab]:
                    scores[lab] += w_bi * lex_bi[lab][bi]


        for d in doms:
            if d in domains_kept:
                for lab in labels:
                    scores[lab] += w_dom * dom_scores[lab].get(d, 0.0)


        if "query" in label_set and "?" in t:
            scores["query"] += w_qmark


        best_lab = max(scores.items(), key=lambda x: x[1])[0]
        if abs(scores[best_lab]) < 1e-6:
            return majority_label
        return best_lab

    y_test_lex2 = np.array([predict_lex_url(txt, major_class) for txt in X_test_text])

    acc_lex2 = accuracy_score(y_test, y_test_lex2)
    print(f"\nAccuracy baseline léxica+URL (mejorada) = {acc_lex2:.4f}")

    print("\nClassification report (TEST) - Baseline léxica+URL (mejorada):")
    print(classification_report(y_test, y_test_lex2, digits=4))

    print("\nEjemplo de predicciones (primeros 20 ejemplos del TEST):")
    for i in range(20):
        print(f"{i}) real={y_test[i]:8s}  pred={y_test_lex2[i]:8s}  text={X_test_text[i][:70]!r}...")


    acc_hybrid=acc_lex2
    # TABLA RESUMEN DE ACCURACIES
    results = {
        "Baseline mayoría (siempre comment)": baseline_acc,
        "Baseline híbrida (léxica + URL)": acc_hybrid,
        "TF-IDF + KNN": acc_test_tfidf,
        "Word2Vec + KNN": acc_test_w2v,
        "SBERT + KNN": acc_test_bert,
        "TF-IDF + CNN": acc_test_tfidf_cnn,
        "Word2Vec + CNN": acc_test_w2v_cnn,
        "SBERT + CNN": acc_test_bert_cnn,
        f"Transformer fine-tuned ({transformer_model_name})": acc_test_transformer,
    }

    df_results = (
        pd.DataFrame.from_dict(results, orient="index", columns=["accuracy"])
          .sort_values("accuracy", ascending=False)
    )

    print("RESUMEN DE ACCURACIES (ordenado de mayor a menor):\n")
    display(df_results)


    # GRÁFICO GLOBAL DE TODOS LOS MODELOS
    plt.figure(figsize=(10, 6))

    plt.barh(df_results.index, df_results["accuracy"])
    plt.xlabel("Accuracy en test")
    plt.title("Comparación global de modelos (accuracy)")
    plt.xlim(0, 1.0)

    for i, v in enumerate(df_results["accuracy"]):
        plt.text(v + 0.01, i, f"{v:.3f}", va="center")

    plt.gca().invert_yaxis()  
    plt.tight_layout()
    plt.show()

    models_knn = {
        "TF-IDF + KNN": acc_test_tfidf,
        "Word2Vec + KNN": acc_test_w2v,
        "SBERT + KNN": acc_test_bert,
    }

    df_knn = pd.DataFrame.from_dict(models_knn, orient="index", columns=["accuracy"])

    plt.figure(figsize=(6, 4))
    plt.bar(df_knn.index, df_knn["accuracy"])
    plt.ylabel("Accuracy en test")
    plt.title("Comparación KNN con distintas representaciones")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=15)

    for i, v in enumerate(df_knn["accuracy"]):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")



    models_cnn = {
        "TF-IDF + CNN": acc_test_tfidf_cnn,
        "Word2Vec + CNN": acc_test_w2v_cnn,
        "SBERT + CNN": acc_test_bert_cnn,
        f"Transformer fine-tuned ({transformer_model_name})": acc_test_transformer,
    }

    df_cnn = pd.DataFrame.from_dict(models_cnn, orient="index", columns=["accuracy"])

    plt.figure(figsize=(7, 4))
    plt.bar(df_cnn.index, df_cnn["accuracy"])
    plt.ylabel("Accuracy en test")
    plt.title("Comparación CNN vs Transformer")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=20)

    for i, v in enumerate(df_cnn["accuracy"]):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

    plt.tight_layout()
    plt.show()



    models_baseline = {
        "Baseline mayoría": baseline_acc,
        "Baseline híbrida": acc_hybrid,
        f"Transformer fine-tuned ({transformer_model_name})": acc_test_transformer,
    }

    df_base = pd.DataFrame.from_dict(models_baseline, orient="index", columns=["accuracy"])

    plt.figure(figsize=(7, 4))
    plt.bar(df_base.index, df_base["accuracy"])
    plt.ylabel("Accuracy en test")
    plt.title("Baselines simbólicas vs Transformer fine-tuned")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=20)

    for i, v in enumerate(df_base["accuracy"]):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

    plt.tight_layout()
    plt.show()



    label_counts = Counter(y_test)
    labels = list(label_counts.keys())
    counts = [label_counts[l] for l in labels]

    plt.figure(figsize=(5, 4))
    plt.bar(labels, counts)
    plt.title("Distribución de clases en el conjunto de test")
    plt.ylabel("Número de ejemplos")
    plt.tight_layout()
    plt.show()

    rep_transf = classification_report(
        y_test_true_labels,
        y_test_pred_labels_transformer,
        output_dict=True
    )

    # Baseline mayoría siempre 'comment'
    y_test_majority = np.array([major_class] * len(y_test))
    rep_majority = classification_report(
        y_test,
        y_test_majority,
        output_dict=True
    )

    labels = label_encoder.classes_  # ['comment', 'deny', 'query', 'support']

    f1_transf = [rep_transf[l]["f1-score"] for l in labels]
    f1_major  = [rep_majority[l]["f1-score"] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(7, 4))
    plt.bar(x - width/2, f1_major,  width, label="Baseline mayoría")
    plt.bar(x + width/2, f1_transf, width, label="Transformer")

    plt.xticks(x, labels)
    plt.ylabel("F1-score")
    plt.title("F1 por clase: baseline vs Transformer")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Etiquetas en el orden correcto
    labels = label_encoder.classes_

    # Matriz de confusión del Transformer
    cm = confusion_matrix(y_test_true_labels,
                          y_test_pred_labels_transformer,
                          labels=labels)

    plt.figure(figsize=(5, 4))
    ax = plt.gca()

    im = ax.imshow(cm)  # sin cmap para no complicar

    # Ejes
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Matriz de confusión – Transformer fine-tuned")

    # Escribir los números dentro de cada celda
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()
