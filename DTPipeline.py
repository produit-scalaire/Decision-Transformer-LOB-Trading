import os
import sys
import time
import argparse
import subprocess

def run_step(command: list, step_name: str):
    """
    Exécute un processus enfant de manière bloquante et vérifie son code de retour.
    L'isolation stricte garantit la libération du cache RAM/VRAM au niveau de l'OS.
    """
    print("=" * 80)
    print(f"[PIPELINE STEP] : {step_name}")
    print(f"Executing       : {' '.join(command)}")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # Popen permet d'afficher la sortie standard en temps réel (stdout pipe)
        process = subprocess.Popen(
            command,
            stdout=sys.stdout,
            stderr=sys.stderr,
            universal_newlines=True
        )
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError(f"L'étape '{step_name}' a échoué avec le code {process.returncode}.")
            
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTION] Arrêt forcé de l'étape '{step_name}'.")
        process.terminate()
        sys.exit(1)
        
    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] '{step_name}' terminée en {elapsed:.2f} secondes.\n")

def main():
    parser = argparse.ArgumentParser(description="Pipeline complet : Data -> Train -> Eval")
    
    # 1. Hyperparamètres Globaux
    parser.add_argument("--dataset", type=str, default="fi2010", choices=["fi2010", "deeplob"], help="Choix du dataset source")
    parser.add_argument("--num_trajectories", type=int, default=50000, help="Taille du support D")
    parser.add_argument("--seq_len", type=int, default=1024, help="Longueur K du contexte")
    parser.add_argument("--data_file", type=str, default="offline_dataset.pt", help="Nom de l'artefact de données")
    
    # 2. Hyperparamètres d'Entraînement
    parser.add_argument("--epochs", type=int, default=15, help="Nombre d'itérations d'optimisation")
    parser.add_argument("--batch_size", type=int, default=128, help="Taille du batch (128-256 opti pour RTX 5090)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning Rate maximal (Cosine Decay)")
    
    # 3. Hyperparamètres du Modèle & Évaluation
    parser.add_argument("--d_model", type=int, default=256, help="Dimension latente")
    parser.add_argument("--n_heads", type=int, default=8, help="Nombre de têtes d'attention")
    parser.add_argument("--n_layers", type=int, default=6, help="Profondeur du réseau")
    parser.add_argument("--target_rtg", type=float, default=2.0, help="Objectif PnL conditionnel pour l'inférence")
    
    # 4. Flags de Contrôle
    parser.add_argument("--skip_data", action="store_true", help="Passe l'étape de génération si les données existent")
    parser.add_argument("--skip_train", action="store_true", help="Passe l'étape d'entraînement si les poids existent")
    
    args = parser.parse_args()
    
    pipeline_start = time.time()
    
    # ---------------------------------------------------------
    # ÉTAPE 1 : GÉNÉRATION DES TRAJECTOIRES (CPU Multithreading)
    # ---------------------------------------------------------
    if not args.skip_data:
        cmd_data = [
            sys.executable, "generate_trajectories.py",
            "--dataset", str(args.dataset),
            "--num_trajectories", str(args.num_trajectories),
            "--seq_len", str(args.seq_len),
            "--output", str(args.data_file)
        ]
        run_step(cmd_data, "Génération des Trajectoires Hors-Ligne")
    else:
        print("[PIPELINE] Étape de génération ignorée (--skip_data).")

    # ---------------------------------------------------------
    # ÉTAPE 2 : ENTRAÎNEMENT DU MODÈLE (GPU RTX 5090)
    # ---------------------------------------------------------
    if not args.skip_train:
        cmd_train = [
            sys.executable, "train.py",
            "--data_path", str(args.data_file),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr)
            # Ajout implicite des dimensions si ton train.py gère ces arguments (à adapter au besoin)
        ]
        run_step(cmd_train, "Optimisation du Decision Transformer")
    else:
        print("[PIPELINE] Étape d'entraînement ignorée (--skip_train).")

    # ---------------------------------------------------------
    # ÉTAPE 3 : ÉVALUATION FINANCIÈRE (GPU Inference Vectorisée)
    # ---------------------------------------------------------
    # Identification automatique du dernier checkpoint généré
    # Format attendu depuis train.py : dt_model_epXX.pt
    expected_model_file = f"dt_model_ep{args.epochs:02d}.pt"
    
    if not os.path.exists(expected_model_file):
        print(f"[ATTENTION] Le poids {expected_model_file} est introuvable. Recherche du dernier checkpoint disponible...")
        checkpoints = [f for f in os.listdir('.') if f.startswith('dt_model_ep') and f.endswith('.pt')]
        if not checkpoints:
            raise FileNotFoundError("Aucun poids de modèle trouvé pour l'évaluation.")
        expected_model_file = sorted(checkpoints)[-1]
        print(f"[INFO] Utilisation de : {expected_model_file}")

    cmd_eval = [
        sys.executable, "evaluate.py",
        "--model_weights", expected_model_file,
        "--test_data", str(args.data_file),
        "--d_model", str(args.d_model),
        "--n_heads", str(args.n_heads),
        "--n_layers", str(args.n_layers),
        "--context_len", str(args.seq_len),
        "--target_rtg", str(args.target_rtg)
    ]
    run_step(cmd_eval, "Évaluation et Déploiement Stochastique")

    # ---------------------------------------------------------
    # BILAN
    # ---------------------------------------------------------
    total_time = time.time() - pipeline_start
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    
    print("=" * 80)
    print(f"PIPELINE COMPLET ACHEVÉ EN {hours}h {minutes}m {seconds:.2f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()