# IMPORT E CREAZIONE ENVIRONMENT

import gymnasium as gym
import math
import random
import matplotlib                   # per i grafici
import matplotlib.pyplot as plt

# namedtuple crea una struttura dati leggibile
# deque è una coda che elimina automaticamente i dati più vecchi quando si riempie (cuore della Replay Memory)
from collections import namedtuple, deque
from itertools import count

# i seguenti "import" rappresentano il nostro motore matematico 
import torch                        # gestisce i tensori 
import torch.nn as nn               # (neural network) contiene i blocchi base per costruire la rete
import torch.optim as optim         # contiene gli algoritmi per aggiornare i pesi 
import torch.nn.functional as F     # contiene funzioni matematiche pure (e.g. ReLU)

# "accendiamo" il simulatore
env = gym.make("CartPole-v1") # useremo "env" per leggere lo stato dei sensori e inviare comandi ai motori. Fa da ponte con il mondo fisico

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion() # è il drawnow di matlab

# if GPU is to be used -> essenzialmenete specifico dove devono risiedere i dati (CPU vs VRAM della GPU)
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else # è il mio caso (Mac)
    "cpu" # in alternativa, usa la CPU 
)


# To ensure reproducibility during training, you can fix the random seeds
# by uncommenting the lines below. This makes the results consistent across
# runs, which is helpful for debugging or comparing different approaches.
# That said, allowing randomness can be beneficial in practice, as it lets
# the model explore different training trajectories.

# seed = 42
# random.seed(seed)
# torch.manual_seed(seed)
# env.reset(seed=seed)
# env.action_space.seed(seed)
# env.observation_space.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)


# REPLAY MEMORY

# creiamo una struct (Transition) che raggruppi le 4 variabili fondamentali di un Markov Decision Process (MDP)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward')) # questa è letteralmente una struct


class ReplayMemory(object):

    def __init__(self, capacity):
        # il seguente comando, se si arriva alla fine di un vettore (maxlen=capacity) elimina automaticamente il primo elemento (vecchio) e scrive il nuovo dato in prima posizione (in sostanza è come se l'indice del vettore ripartisse)
        self.memory = deque([], maxlen=capacity) 

    def push(self, *args):
        # questa funzione prende i 4 valori di un'esperienza e li "impacchetta" nella struct Transition, infilandoli poi nel buffer
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # invece di restituire gli ultimi dati inseriti, questa funzione usa random.sample per pescare un "pugno" di esperienze (un batch) in modo completamente casuale da tutto il buffer
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    


# ALGORITMO DQN 

class DQN(nn.Module): # questa classe è letteralmente il cervello del nostro sistema 

    def __init__(self, n_observations, n_actions):
        # Stiamo creando una classe DQN che eredita da nn.Module. In PyTorch, nn.Module è la classe base per tutte le reti neurali. Ereditando da essa, la tua classe ottiene automaticamente i "superpoteri" di PyTorch, come la capacità di tenere traccia dei pesi da aggiornare e di spostare la rete sulla GPU M1.
        # Costruttore (__init__): Chiediamo in ingresso n_observations (quanti sensori ha il robot, o nel caso del CartPole, 4 valori) e n_actions (quante azioni può compiere, per il CartPole sono 2).
        super(DQN, self).__init__()

        # nn.Linear: è un layer "linear", esegue una semplice operazione matriciale Y = X*W' + b, ovvero: moltiplica l'input X per una matrice dei pesi W e aggiunge un vettore di bias b
        # creiamo una rete a 3 strati (3 layers)
        self.layer1 = nn.Linear(n_observations, 128) # layer di input che mappa i sensori verso 128 neuroni nascosti
        self.layer2 = nn.Linear(128, 128) # layer nascosto da 128 neuroni
        self.layer3 = nn.Linear(128, n_actions) # layer di output che mappa i 128 neuroni verso le possibili azioni

    # Called with either one element (one state) to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]) -> ovvero non restituisce un "vai a dx/sx", ma i Q-values (che qui chiama exp, ovvero Expected Return)

    # Notes:
    # - x è un tensore (vettore) che contiene i valori correnti dei sensori
    # - F.Relu(...) è la funzione di attivazione -> introduce non linearità (perché? perché il mondo reale è non lineare, perciò se vogliamo approssimare una funzione non lineare non possiamo usare una semplice funzione lineare)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x) # ritorno i Q values (non c'è ReLU perché possono anche essere valori negativi)
    



### TRAINING

# HYPERPARAMETERS AND UTILITIES

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4


# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device) # creo rete di training (actor)
target_net = DQN(n_observations, n_actions).to(device) # creo rete target (critic)
target_net.load_state_dict(policy_net.state_dict()) # forzo critic ad avere gli stessi pesi di actor

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state): # implementazione eps-greedy 
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY) # creo la soglia per l'eps-greedy
    steps_done += 1
    if sample > eps_threshold: # se il numero casuale è maggiore della soglia => faccio EXPLOITATION
        with torch.no_grad(): # faccio le cose senza calcolare derivate (sto solo decidendo i pesi) -> risparmio tempo 
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else: # faccio EXPLORATION
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []

# PLOT 

def plot_durations(show_result=False): 
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())




# OTTIMIZZAZIONE

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE) # pesca a caso BATCH_SIZE ricordi 
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    # lo faccio perché non voglio BATCH_SIZE oggetti separati ma un unico grande tensore di BATCH_SIZE stati, azioni, etc

    batch = Transition(*zip(*transitions)) 

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    # GEMINI per capire cosa fa (cercare mettendo: Quando il robot sbatte contro un ostacolo nell'ambiente simulato, il gioco finisce (terminated = True). In quel preciso istante, il codice del tutorial decide di salvare la transizione nel Replay Buffer inserendo come stato futuro un valore nullo: next_state = None (l'equivalente di un puntatore NULL in C).)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    
    # prendo solo gli stati validi e li incollo in un unico grande tensore (in altre parole, scarto gli incidenti)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None]) 
    state_batch = torch.cat(batch.state) # concateno stati
    action_batch = torch.cat(batch.action) # concateno azioni
    reward_batch = torch.cat(batch.reward) # concateno rewards

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # .gather(1, action_batch) serve a pescare solo la colonna corrispondente all'azione presa
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device) # creo vettore stati futuri
    with torch.no_grad():
        # IMPLEMENTO GIò la DDQN (commentata la parte della DQN)
        #next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values # calcoliamo il valore futuro e lo piazziamo nel vettore creato (ma solo degli stati "puliti", non terminali)

        # 1. LA SELEZIONE (Policy Net): Chiediamo alla rete in addestramento l'indice dell'azione migliore
        best_actions = policy_net(non_final_next_states).max(1).indices.unsqueeze(1)
    
        # 2. LA VALUTAZIONE (Target Net): Chiediamo alla rete target di valutare SOLO quell'azione specifica
        next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, best_actions).squeeze(1)
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    # Confrontiamo la predizione passata (state_action_values) con il target appena calcolato. Usiamo la Huber Loss (SmoothL1Loss) invece del classico Errore Quadratico Medio (MSE). La Huber Loss è quadratica per errori piccoli, ma diventa lineare per errori grandi. Questo impedisce alla rete di "impazzire" (i gradienti esplodono) quando riceve una ricompensa molto anomala.
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad() # azzera i gradienti calcolati al passo precedente 
    loss.backward() # Il cuore dell'Autograd. PyTorch ripercorre a ritroso tutto il grafo computazionale e calcola le derivate parziali dell'errore rispetto a ognuno dei 300 neuroni della tua rete. Tutto in un millisecondo.

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100) # Un sistema di sicurezza. Taglia i gradienti se superano il valore 100 per evitare sbalzi troppo violenti nell'apprendimento.

    optimizer.step() # Applica l'algoritmo di ottimizzazione (AdamW) e aggiorna effettivamente i pesi matematici della rete neurale. In questo momento, il tuo robot ha ufficialmente imparato qualcosa di nuovo!



if torch.cuda.is_available() or torch.backends.mps.is_available(): # sceglie num episodi in base a se la GPU è attiva
    num_episodes = 600
else:
    num_episodes = 50 # 50 episodi se uso la CPU 

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset() # carrello messo al centro, pronto a cadere. inizia la partita
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) # unsquese trasforma il vettore in una matrice

    for t in count(): # contatore infinio -> mi fermo solo quando il carrello cade (break)
        action = select_action(state) # eps-greedy
        observation, reward, terminated, truncated, _ = env.step(action.item()) # ambiente avanza di un frame
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None # se c'è stato un incidente il futuro non esiste
        else:
            # se non c'è stato un incidente pendiamo la nuova lettura dei sensori (observation) e la trasformiamo in una matrice 1x4 con l' .unsqueeze(0) pronto per la rete
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights -> mescolo i pesi della training net nella target net
        # TAU regola quanto velocemente mischiamo i pesi
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done: # se il bastone cade
            episode_durations.append(t + 1) # segniamo quanto è durato l'episodio 
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()