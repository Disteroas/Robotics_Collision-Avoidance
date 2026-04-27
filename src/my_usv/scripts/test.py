#!/usr/bin/env python3
import rclpy
import torch
import os
import numpy as np

# Ora le importazioni funzionano nativamente senza trucchi!
from usv_env import UsvEnv
from ddqn_model import DDQN

def main(args=None):
    rclpy.init(args=args)
    
    print("\n" + "="*50)
    print("  USV AUTONOMOUS NAVIGATION TEST")
    print("="*50 + "\n")
    
    env = UsvEnv()
    
    # Cerca il file nella stessa cartella in cui si trova test.py
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, 'best_ddqn_model.pth')
    
    if not os.path.exists(model_path):
        print(f"\n[ERRORE] Cervello '{model_path}' non trovato!")
        env.destroy_node()
        rclpy.shutdown()
        return
        
    # Carica la rete neurale
    q_net = DDQN()
    q_net.load_state_dict(torch.load(model_path))
    q_net.eval()  # Disabilita il training
    
    print(f"Cervello '{model_path}' caricato con successo!")
    print("Inizio navigazione autonoma (Epsilon = 0.0). Premi Ctrl+C per fermare.\n")
    
    try:
        for episode in range(1, 4):  
            state = env.reset_environment()
            episode_reward = 0.0
            crashed = False
            steps = 0
            
            print(f"--- Partenza Test {episode} ---")
            
            while steps < 1000:
                state_t = torch.FloatTensor(state).unsqueeze(0)
                
                with torch.no_grad():
                    action_idx = int(q_net(state_t).argmax(dim=1).item())
                    
                next_state, reward, done = env.step_action(action_idx)
                
                episode_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    crashed = True
                    break
                    
            if crashed:
                print(f"Test {episode} concluso: 💥 COLLISIONE dopo {steps} step (Reward: {episode_reward:.1f})\n")
            else:
                print(f"Test {episode} concluso: ✅ SUCCESSO! Navigazione per {steps} step (Reward: {episode_reward:.1f})\n")
                
    except KeyboardInterrupt:
        print("\nTest interrotto manualmente.")
    finally:
        env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()