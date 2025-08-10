import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# Configure the page
st.set_page_config(page_title="Botnet Detection", layout="wide")

# Title
st.title("🤖 Botnet Detection in Robotic Network Logs")

# Load data function
@st.cache_data
def load_data():
    """Load the dataset with error handling"""
    try:
        # Try different possible file paths
        possible_paths = [
            "synthetic_robot_logs.csv",
            "data/synthetic_robot_logs.csv",
            "./synthetic_robot_logs.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                st.success(f"✅ Dataset loaded from {path}")
                return df
        
        # If no file found, create sample data for demonstration
        st.warning("⚠️ Dataset file not found. Using sample data for demonstration.")
        return create_sample_data()
        
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration when real data is not available"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate normal traffic
    normal_packet_sizes = np.random.normal(500, 150, n_samples//2)
    normal_intervals = np.random.exponential(2.0, n_samples//2)
    
    # Generate botnet traffic (different patterns)
    botnet_packet_sizes = np.random.normal(800, 100, n_samples//2)
    botnet_intervals = np.random.exponential(0.5, n_samples//2)
    
    # Combine data
    packet_sizes = np.concatenate([normal_packet_sizes, botnet_packet_sizes])
    intervals = np.concatenate([normal_intervals, botnet_intervals])
    labels = np.concatenate([np.zeros(n_samples//2), np.ones(n_samples//2)])
    
    # Ensure positive values
    packet_sizes = np.abs(packet_sizes)
    intervals = np.abs(intervals)
    
    df = pd.DataFrame({
        'packet_size': packet_sizes,
        'interval': intervals,
        'is_botnet': labels.astype(int)
    })
    
    return df.sample(frac=1).reset_index(drop=True)  # Shuffle

# Train scikit-learn model
@st.cache_resource
def train_model():
    """Train a scikit-learn model for botnet detection"""
    try:
        # Load data for training
        df = load_data()
        if df is not None and 'is_botnet' in df.columns:
            X = df[['packet_size', 'interval']].values
            y = df['is_botnet'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
            model.fit(X_train_scaled, y_train)
            
            # Calculate accuracy
            train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
            test_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
            
            st.success(f"✅ Model trained successfully!")
            st.info(f"📊 Training Accuracy: {train_accuracy:.3f} | Test Accuracy: {test_accuracy:.3f}")
            
            return model, scaler, test_accuracy
        
        else:
            st.error("❌ Unable to train model: Invalid dataset")
            return None, None, 0.0
            
    except Exception as e:
        st.error(f"❌ Error training model: {e}")
        return None, None, 0.0

# Load data and train model
df = load_data()
model, scaler, model_accuracy = train_model()

if df is not None and model is not None:
    # Display basic info about the dataset
    st.subheader("📊 Dataset Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    
    with col2:
        if 'is_botnet' in df.columns:
            botnet_count = df['is_botnet'].sum()
            st.metric("Botnet Records", int(botnet_count))
    
    with col3:
        if 'is_botnet' in df.columns:
            normal_count = len(df) - df['is_botnet'].sum()
            st.metric("Normal Records", int(normal_count))
    
    with col4:
        st.metric("Model Accuracy", f"{model_accuracy:.1%}")
    
    # Dataset preview
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(10))
    
    # Check if required columns exist
    required_columns = ['packet_size', 'interval']
    if all(col in df.columns for col in required_columns):
        
        # User input for prediction
        st.subheader("🔍 Predict Botnet Activity")
        
        # Get reasonable ranges from data
        min_packet = float(df['packet_size'].min())
        max_packet = float(df['packet_size'].max())
        mean_packet = float(df['packet_size'].mean())
        
        min_interval = float(df['interval'].min())
        max_interval = float(df['interval'].max())
        mean_interval = float(df['interval'].mean())
        
        col1, col2 = st.columns(2)
        with col1:
            packet_size = st.number_input(
                f"Packet Size (Range: {min_packet:.1f} - {max_packet:.1f})", 
                min_value=0.0, 
                max_value=max_packet * 2,
                value=mean_packet,
                step=10.0,
                help="Size of the network packet in bytes"
            )
        
        with col2:
            interval = st.number_input(
                f"Interval (Range: {min_interval:.2f} - {max_interval:.2f})", 
                min_value=0.0, 
                max_value=max_interval * 2,
                value=mean_interval,
                step=0.1,
                help="Time interval between packets in seconds"
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔮 Predict", type="primary", use_container_width=True):
                try:
                    # Prepare input data
                    input_data = np.array([[packet_size, interval]])
                    input_scaled = scaler.transform(input_data)
                    
                    # Get prediction and probability
                    prediction = model.predict(input_scaled)[0]
                    probability = model.predict_proba(input_scaled)[0]
                    confidence = max(probability)
                    
                    # Display result with styling
                    if prediction == 1:
                        st.error(f"🔴 **Botnet Activity Detected!**")
                        st.error(f"Confidence: {confidence:.1%}")
                    else:
                        st.success(f"🟢 **Normal Activity**")
                        st.success(f"Confidence: {confidence:.1%}")
                    
                    # Show additional details
                    with st.expander("📊 Prediction Details"):
                        st.write(f"**Prediction:** {'Botnet' if prediction == 1 else 'Normal'}")
                        st.write(f"**Normal Probability:** {probability[0]:.4f}")
                        st.write(f"**Botnet Probability:** {probability[1]:.4f}")
                        st.write(f"**Model Accuracy:** {model_accuracy:.1%}")
                        st.write(f"**Input values:** Packet Size = {packet_size:.2f}, Interval = {interval:.2f}")
                        
                        # Show how this compares to dataset
                        percentile_packet = (df['packet_size'] < packet_size).mean() * 100
                        percentile_interval = (df['interval'] < interval).mean() * 100
                        st.write(f"**Data percentiles:** Packet size = {percentile_packet:.1f}th, Interval = {percentile_interval:.1f}th")
                        
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
        
        with col2:
            if st.button("🎲 Random Sample", use_container_width=True):
                # Select a random sample from the dataset
                sample = df.sample(1).iloc[0]
                st.session_state.packet_size = float(sample['packet_size'])
                st.session_state.interval = float(sample['interval'])
                st.rerun()
        
        # Visualization section
        st.subheader("📊 Data Visualization")
        
        if 'is_botnet' in df.columns and len(df) > 0:
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["🎯 Scatter Plot", "📈 Distributions", "⚖️ Class Balance"])
            
            with tab1:
                st.write("**Packet Size vs Interval Analysis**")
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Create scatter plot with better styling
                colors = ['#2E8B57', '#DC143C']  # Sea green and crimson
                labels = ['Normal', 'Botnet']
                
                for i, (label, color) in enumerate(zip(labels, colors)):
                    mask = df['is_botnet'] == i
                    if mask.any():
                        ax.scatter(df[mask]['packet_size'], df[mask]['interval'], 
                                 alpha=0.6, s=50, c=color, label=label, edgecolors='white', linewidth=0.5)
                
                ax.set_xlabel("Packet Size (bytes)", fontsize=12)
                ax.set_ylabel("Interval (seconds)", fontsize=12)
                ax.set_title("Network Traffic Patterns: Botnet vs Normal Activity", fontsize=14, fontweight='bold')
                ax.legend(title='Activity Type', title_fontsize=10)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Add interpretation
                st.info("""
                **Interpretation Guide:**
                - 🟢 **Green points (Normal):** Typical network behavior
                - 🔴 **Red points (Botnet):** Potentially malicious activity
                - Look for clustering patterns that distinguish botnet from normal traffic
                """)
            
            with tab2:
                st.write("**Feature Distribution Analysis**")
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # Packet size distribution
                for class_val, color, label in zip([0, 1], ['green', 'red'], ['Normal', 'Botnet']):
                    subset = df[df['is_botnet'] == class_val]
                    if len(subset) > 0:
                        axes[0, 0].hist(subset['packet_size'], alpha=0.7, color=color,
                                       label=label, bins=30, edgecolor='black', linewidth=0.5)
                axes[0, 0].set_xlabel("Packet Size (bytes)")
                axes[0, 0].set_ylabel("Frequency")
                axes[0, 0].set_title("Packet Size Distribution")
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                # Interval distribution
                for class_val, color, label in zip([0, 1], ['green', 'red'], ['Normal', 'Botnet']):
                    subset = df[df['is_botnet'] == class_val]
                    if len(subset) > 0:
                        axes[0, 1].hist(subset['interval'], alpha=0.7, color=color,
                                       label=label, bins=30, edgecolor='black', linewidth=0.5)
                axes[0, 1].set_xlabel("Interval (seconds)")
                axes[0, 1].set_ylabel("Frequency")
                axes[0, 1].set_title("Interval Distribution")
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                # Box plots
                try:
                    df_melted = df.melt(id_vars=['is_botnet'], value_vars=['packet_size'], 
                                       var_name='feature', value_name='value')
                    sns.boxplot(data=df_melted, x='is_botnet', y='value', ax=axes[1, 0])
                    axes[1, 0].set_xlabel("Activity Type (0=Normal, 1=Botnet)")
                    axes[1, 0].set_ylabel("Packet Size")
                    axes[1, 0].set_title("Packet Size by Activity Type")
                    
                    df_melted2 = df.melt(id_vars=['is_botnet'], value_vars=['interval'], 
                                        var_name='feature', value_name='value')
                    sns.boxplot(data=df_melted2, x='is_botnet', y='value', ax=axes[1, 1])
                    axes[1, 1].set_xlabel("Activity Type (0=Normal, 1=Botnet)")
                    axes[1, 1].set_ylabel("Interval")
                    axes[1, 1].set_title("Interval by Activity Type")
                except Exception as e:
                    st.warning(f"Could not create box plots: {e}")
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with tab3:
                st.write("**Class Distribution Analysis**")
                
                class_counts = df['is_botnet'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Enhanced pie chart
                    fig, ax = plt.subplots(figsize=(8, 8))
                    colors = ['#98FB98', '#FFB6C1']  # Light green and light pink
                    wedges, texts, autotexts = ax.pie(class_counts.values, labels=['Normal', 'Botnet'], 
                                                     autopct='%1.1f%%', colors=colors, startangle=90,
                                                     explode=(0.05, 0.05), shadow=True)
                    
                    # Enhance text
                    for autotext in autotexts:
                        autotext.set_color('black')
                        autotext.set_fontweight('bold')
                    
                    ax.set_title("Class Distribution", fontsize=16, fontweight='bold', pad=20)
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    # Enhanced bar chart
                    fig, ax = plt.subplots(figsize=(8, 8))
                    bars = ax.bar(['Normal', 'Botnet'], class_counts.values, 
                                 color=['#98FB98', '#FFB6C1'], edgecolor='black', linewidth=1.5)
                    ax.set_ylabel("Count", fontsize=12)
                    ax.set_title("Activity Type Counts", fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
                    
                    st.pyplot(fig)
                    plt.close()
                
                # Add statistics
                st.subheader("📈 Dataset Statistics")
                stats_col1, stats_col2 = st.columns(2)
                
                with stats_col1:
                    st.metric("Class Balance Ratio", f"{class_counts[0]/class_counts[1]:.2f}:1")
                    st.metric("Majority Class", f"{(class_counts.max()/len(df)*100):.1f}%")
                
                with stats_col2:
                    st.metric("Dataset Size", f"{len(df):,} samples")
                    st.metric("Feature Count", len(required_columns))
        
        else:
            st.info("ℹ️ Visualization requires 'is_botnet' column in the dataset")
    
    else:
        st.error(f"❌ Required columns missing. Expected: {required_columns}, Found: {list(df.columns)}")

else:
    st.error("❌ Unable to load dataset or train model")

# Sidebar with enhanced information
st.sidebar.header("ℹ️ About This App")
st.sidebar.write("""
This application demonstrates **botnet detection** in robotic network logs using machine learning techniques.

**Key Features:**
- 📊 Interactive data exploration
- 🔮 Real-time botnet prediction  
- 📈 Comprehensive visualizations
- 🎯 Statistical analysis
- 🤖 Random Forest classifier
""")

st.sidebar.header("🚀 How It Works")
st.sidebar.write("""
1. **Data Generation:** Creates synthetic network traffic data
2. **Feature Engineering:** Standardizes packet size and interval features
3. **Model Training:** Uses Random Forest for classification
4. **Prediction:** Classifies new traffic as normal or botnet
5. **Visualization:** Shows data patterns and model performance
""")

st.sidebar.header("📁 File Support")
st.sidebar.write("""
**Supported Files:**
- `synthetic_robot_logs.csv` - Custom dataset (optional)

**Required Columns:**
- `packet_size` - Network packet size in bytes  
- `interval` - Time between packets in seconds
- `is_botnet` - Binary label (0=Normal, 1=Botnet)

**Note:** App generates sample data if no file is provided!
""")

st.sidebar.header("🔧 Technical Details")
if df is not None and model is not None:
    st.sidebar.success(f"""
    **Current Setup:**
    ✅ Model: Random Forest  
    ✅ Records: {len(df):,}  
    ✅ Features: {', '.join(required_columns)}  
    ✅ Accuracy: {model_accuracy:.1%}
    """)
else:
    st.sidebar.error("❌ Setup incomplete")

# Performance metrics section
if model is not None and df is not None:
    with st.expander("📊 Model Performance Details"):
        st.subheader("Model Configuration")
        st.code(f"""
Random Forest Classifier:
- Estimators: 100
- Max Depth: 10
- Min Samples Split: 5
- Random State: 42
- Test Accuracy: {model_accuracy:.3f}
        """)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': required_columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(importance_df['Feature'], importance_df['Importance'], 
                         color='skyblue', edgecolor='navy', linewidth=1.5)
            ax.set_xlabel("Features")
            ax.set_ylabel("Importance")
            ax.set_title("Feature Importance in Botnet Detection")
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close() 