# Intelligent Connection-Type Selection using Reinforcement Learning

This document outlines the architecture and implementation of a Reinforcement Learning (RL) agent designed to intelligently choose the optimal connection type (`P2P` or `Server`) for a communication session based on real-time network conditions.

The system is designed not only to make smart decisions from day one but also to learn and adapt over time based on user feedback.

## The Core Idea: An Agent That Learns from Experience

At its heart, this system is an "agent" that we've trained to play a simple game: for a given set of network conditions, choose the action (P2P or Server) that will result in the highest user satisfaction.

It learns this through three key concepts:
*   **State (S):** A snapshot of the current network conditions (e.g., `low latency, no packet loss, high bandwidth, on wifi`).
*   **Action (A):** The choice the agent can make: `0` for P2P or `1` for Server.
*   **Reward (R):** The outcome of taking an action in a state, which we get from a user's call quality rating (1-5 stars, normalized to a 0.25-1.0 score).

The goal of the training process is to produce a "cheat sheet," called a **Q-Table**, that tells the agent the expected reward for any given (State, Action) pair.

## Phase 1: How the Model Was Trained

The brain of our agent, the Q-Table, was built in a two-step process in a Python environment.

### 1. High-Quality Data Generation

A model is only as good as its data. Instead of relying on messy, incomplete real-world data, we generated a high-quality, synthetic dataset to serve as the "perfect textbook" for our agent. This data was designed to teach the model specific, logical lessons:

*   **P2P is the Hero in Ideal Conditions:** In scenarios with very low latency, zero packet loss, and high bandwidth, the data shows that choosing `p2p` results in an `Excellent` (1.0) reward.
*   **Server is the Hero in Unstable Conditions:** In scenarios with high latency, packet loss, or low bandwidth, the data shows that `p2p` would result in a `Poor` (0.25) reward, while a `server` connection can salvage the call and provide a `Fair` (0.5) or `Good` (0.75) reward.
*   **Server is a Reliable Backup:** In ideal conditions where P2P is `Excellent`, the data shows the server is merely `Good`. This teaches the model that there's a slight performance overhead to using a server, making P2P the preferable choice when possible.

This generated data provides a robust and logical foundation for the agent's knowledge.

### 2. The Reinforcement Learning (Q-Learning) Process

We fed the synthetic dataset to a **Q-learning algorithm**. The algorithm iterates through every row of the data thousands of times. For each row, it looks at the `State`, the `Action` taken, and the `Reward` received, and it updates its Q-Table with the formula:

`New_Q_Value = Old_Q_Value + LearningRate * (Actual_Reward - Old_Q_Value)`

Over many iterations, the Q-values in the table converge, representing a learned "wisdom" about the expected outcome of every choice.

The final output of this entire process is a single file: **`q_table_model.json`**. This file contains everything the application needs.

## Phase 2: In-App Implementation (TypeScript)

To use the model in your TypeScript application, you will need to implement the decision-making logic that uses the `q_table_model.json` file.

### The Model File: `q_table_model.json`

This file is the packaged brain of your agent. It has two main parts:

1.  `bin_edges`: These are the numerical thresholds for `latency`, `loss`, and `bandwidth`. They are **essential** for converting live network stats into the correct state string.
2.  `q_table`: This is a dictionary where keys are state strings (e.g., `"lat_0_loss_0_bw_3_wifi"`) and values are the learned Q-values for P2P (`"0"`) and Server (`"1"`).

### Core Logic in TypeScript

Here is a complete, production-ready function for your app. You should bundle the `q_table_model.json` file as an asset with your application and load it into memory.

```typescript
// --- Step 1: Define a type for your loaded model data ---
interface ModelData {
    bin_edges: {
        latency: number[];
        loss: number[];
        bandwidth: number[];
    };
    q_table: {
        [state: string]: {
            '0': number; // P2P Q-value
            '1': number; // Server Q-value
        };
    };
}

// --- Step 2: Implement the robust recommendation logic ---

/**
 * Finds the correct bin index for a given metric based on the model's bin edges.
 * Handles out-of-range values by clipping them to the known min/max.
 * @param value The live network metric value.
 * @param edges The array of bin edges from the model file.
 * @returns The index of the bin the value falls into.
 */
function findBinIndex(value: number, edges: number[]): number {
    // Gracefully handle inputs outside the training range
    const clippedValue = Math.max(edges[0], Math.min(value, edges[edges.length - 1]));

    // Find the first edge that is greater than the value
    for (let i = 1; i < edges.length; i++) {
        if (clippedValue <= edges[i]) {
            return i - 1;
        }
    }
    return edges.length - 2; // Should not be reached due to clipping
}

/**
 * Recommends a connection type using the trained RL model.
 * @param model The loaded q_table_model.json data.
 * @param latencyMs Current network latency.
 * @param packetLoss Current packet loss percentage.
 * @param bandwidthMbps Current available bandwidth.
 * @param connType 'wifi' or 'cellular'.
 * @returns 'p2p' or 'server'.
 */
function getRecommendation(
    model: ModelData,
    latencyMs: number,
    packetLoss: number,
    bandwidthMbps: number,
    connType: 'wifi' | 'cellular'
): 'p2p' | 'server' {

    // 1. Convert live stats into bin indices
    const latIndex = findBinIndex(latencyMs, model.bin_edges.latency);
    const lossIndex = findBinIndex(packetLoss, model.bin_edges.loss);
    const bwIndex = findBinIndex(bandwidthMbps, model.bin_edges.bandwidth);

    // 2. Construct the state string to match the Q-Table keys
    const state = `lat_${latIndex}_loss_${lossIndex}_bw_${bwIndex}_${connType}`;

    // 3. Look up the state and make a decision
    if (state in model.q_table) {
        const qValues = model.q_table[state];
        const p2pQ = qValues['0'];
        const serverQ = qValues['1'];

        // 4. Apply smart tie-breaking logic for ideal conditions
        const isIdealState = latIndex === 0 && lossIndex === 0;
        const isTie = Math.abs(p2pQ - serverQ) < 0.05;

        if (isIdealState && isTie) {
            return 'p2p'; // Prefer P2P on a tie in ideal conditions
        }
        
        return p2pQ > serverQ ? 'p2p' : 'server';
    } else {
        // 5. If the state is completely new, fall back to the safest option.
        return 'server';
    }
}

// --- Step 3: Example Usage in Your Application ---
/*
// Assume 'modelData' is loaded from your JSON asset
const modelData: ModelData = JSON.parse(loadAsset('q_table_model.json'));

// Get live network stats before starting a call
const liveStats = getLiveNetworkStats(); // { latency: 45, loss: 0.1, ... }

const recommendedAction = getRecommendation(
    modelData,
    liveStats.latency,
    liveStats.packetLoss,
    liveStats.bandwidth,
    liveStats.type
);

console.log(`Model recommends: ${recommendedAction.toUpperCase()}`);

if (recommendedAction === 'p2p') {
    // Initiate P2P connection logic
} else {
    // Initiate Server connection logic
}
*/
```

## Phase 3: The Live Feedback & Retraining Loop

To ensure the model improves and adapts to your users' real-world conditions, you must implement a feedback loop. This is a continuous cycle of logging data, retraining the model, and updating the app.

### Step 1: The App's Role (Logging)

After a session ends, the app must do two things:
1.  **Prompt for a Rating:** Ask the user, "How was your call quality?" (1 to 5 stars).
2.  **Send Data to Your Backend:** Make an API call to a logging endpoint on your server.

The log payload is crucial. It must contain the **initial network state**, the **action taken**, and the **user's rating**.

**Example Log (POST to `https://yourapi.com/log_session`):**
```json
{
  "timestamp": "2025-07-29T10:00:00Z",
  "latency_ms": 70,
  "packetLoss_percent": 0.5,
  "bandwidth_mbps": 50,
  "type": "wifi",
  "action_taken": "server",
  "user_rating": 5
}
```

### Step 2: The Backend's Role (Retraining)

This is an automated process that runs periodically (e.g., weekly or monthly) on your server.
1.  **Fetch New Data:** The script queries your database for all new session logs.
2.  **Map Ratings:** Convert the 1-5 star `user_rating` to our reward scale (`5 -> 1.0`, `4 -> 0.75`, etc.).
3.  **Combine Datasets:** Append this new real-world data to our original `smarter_synthetic_data.csv`. The synthetic data acts as a stable foundation, while the new data provides real-world corrections.
4.  **Re-run Training:** Execute the Python training script on this combined dataset. This will generate a new, even smarter Q-Table and bin edges.
5.  **Export & Version:** Save the output as a new, versioned model file, e.g., `q_table_model_v2.json`.

### Step 3: The App's Role (Model Updates)

The app should not have the model file hardcoded. It should fetch the latest version from your server.
1.  **Create an Endpoint:** Set up a simple endpoint on your server, e.g., `https://yourapi.com/latest_model`, that returns the latest model JSON file.
2.  **Fetch and Cache:** On app launch (or periodically), the app calls this endpoint. If a new version is available, it downloads and saves it to local storage, overwriting the old model.
3.  **Use Local Model:** The `getRecommendation` function should always load the model from this local storage location.

## The Full Flow: A Virtuous Cycle

1.  **Decision:** The app uses its local `q_table_model.json` to choose the best connection type (P2P/Server).
2.  **Experience:** The user has a session.
3.  **Feedback:** The app logs the network conditions and the user's quality rating to your server.
4.  **Learning:** Your backend periodically uses this new feedback to retrain and improve the model.
5.  **Update:** The app downloads the latest, smartest version of the model.

This creates a system that is not only intelligent from day one but also adapts and improves with every single user interaction.