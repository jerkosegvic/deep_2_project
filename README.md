# Nadopunjavanje slike korištenjem difuzijskih modela	
# README

## Pokretanje skripte

Za pokretanje skripte koristite sljedeću naredbu u terminalu:

```sh
python run.py
```

Nakon pokretanja, automatski će se preuzeti **MNIST dataset**.

## Korištenje

Postoje dvije mogućnosti:
1. **Treniranje modela**
2. **Učitavanje već postojećeg modela**

### Treniranje modela
Ako želite trenirati model, odkomentirajte sljedeću liniju koda u skripti:

```python
train.train(model, dataloader, optimizer, scheduler, device, epochs, log_freq, ema_decay)
```

### Učitavanje već istreniranog modela
Ako želite učitati prethodno spremljeni model, skripta će automatski pokušati učitati model iz direktorija `checkpoints`.

```python
checkpoint_path = "checkpoints/epoch_0.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print("Checkpoint učitan uspješno.")
else:
    print(f"Checkpoint nije pronađen na {checkpoint_path}. Molimo pokrenite treniranje prvo.")
    return
```

### Spremljeni modeli
Ako želite učitati određeni model, jednostavno ga stavite u direktorij `checkpoints`. Skripta će automatski provjeriti postoji li model u tom direktoriju i učitati ga ako je dostupan.

---



