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
import time
import plotly.graph_objects as go
import plotly.express as px

# Import the simulator
try:
    from network_simulator import run_enhanced_simulation, NetworkSimulator
    SIMULATOR_AVAILABLE = True
except ImportError:
    SIMULATOR_AVAILABLE = False
    st.warning("⚠️ SimPy network simulator not available. Install simpy to enable simulation features.")

# Configure the page
st.set_page_config(page_title="Botnet Detection with Network Simulation", layout="wide")

# Title
st.title("🤖 Advanced Botnet Detection with Network Simulation")
st.markdown("*Real-time network traffic simulation using SimPy for enhanced botnet detection*")

# Sidebar for simulation controls
if SIMULATOR_AVAILABLE:
    st.sidebar.header("🎮 Network Simulation Controls")
    
    # Simulation parameters
    num_normal = st.sidebar.slider("Normal Robots", min_value=10, max_value=100, value=30, step=5)
    num_compromised = st.sidebar.slider("Compromised Robots", min_value=2, max_value=20, value=8, step=1)
    sim_time = st.sidebar.slider("Simulation Time (seconds)", min_value=30, max_value=600, value=180, step=30)
    
    # Real-time simulation toggle
    real_time_sim = st.sidebar.checkbox("🔴 Real-time Simulation", help="Show live network traffic as it's generated")
    
    # Simulation button
    if st.sidebar.button("🚀 Run New Simulation", type="primary"):
        st.session_state.run_simulation = True
        st.session_state.simulation_data = None

# Load data function with simulation support
@st.cache_data
def load_data(force_simulation=False):
    """Load the dataset with simulation support"""
    try:
        # Check for simulation data first
        if force_simulation and SIMULATOR_AVAILABLE:
            with st.spinner("🔄 Running network simulation..."):
                # Run simulation
                data, stats = run_enhanced_simulation(
                    output_file="simulated_robot_logs.csv",
                    num_normal=30,
                    num_compromised=8,
                    sim_time=180.0
                )
                
            st.success("✅ Network simulation completed!")
            return data
            
        # Try different possible file paths
        possible_paths = [
            "simulated_robot_logs.csv",
            "synthetic_robot_logs.csv",
            "data/synthetic_robot_logs.csv",
            "./synthetic_robot_logs.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                if 'timestamp' in df.columns:
                    st.success(f"✅ Simulation dataset loaded from {path}")
                else:
                    st.success(f"✅ Dataset loaded from {path}")
                return df
        
        # If no file found and simulator available, create simulation data
        if SIMULATOR_AVAILABLE:
            with st.spinner("🔄 No existing data found. Running network simulation..."):
                data, stats = run_enhanced_simulation(
                    output_file="simulated_robot_logs.csv",
                    num_normal=30,
                    num_compromised=8,
                    sim_time=180.0
                )
            return data
        else:
            # Create sample data for demonstration
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

# Real-time simulation function
def run_realtime_simulation():
    """Run real-time simulation with live updates"""
    if not SIMULATOR_AVAILABLE:
        st.error("SimPy simulator not available")
        return
    
    # Create placeholders for real-time updates
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    # Initialize simulator
    simulator = NetworkSimulator(
        num_normal_robots=num_normal,
        num_compromised_robots=num_compromised,
        simulation_time=sim_time
    )
    
    # Setup network
    simulator.setup_network()
    
    # Run simulation with periodic updates
    update_interval = max(5, sim_time // 20)  # Update 20 times during simulation
    packets_data = []
    
    for step in range(0, int(sim_time), update_interval):
        # Run simulation for this step
        simulator.env.run(until=min(step + update_interval, sim_time))
        
        # Collect packets
        current_packets = []
        for robot in simulator.robots:
            current_packets.extend(robot.packets_sent)
        
        if current_packets:
            # Convert to DataFrame
            df_current = simulator._packets_to_dataframe()
            
            # Update status
            with status_placeholder.container():
                st.info(f"🔄 Simulation Progress: {min(step + update_interval, sim_time):.0f}/{sim_time:.0f} seconds")
            
            # Update metrics
            with metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Packets", len(df_current))
                with col2:
                    st.metric("Normal Traffic", len(df_current[df_current['is_botnet'] == 0]))
                with col3:
                    st.metric("Botnet Traffic", len(df_current[df_current['is_botnet'] == 1]))
                with col4:
                    st.metric("Botnet Ratio", f"{df_current['is_botnet'].mean():.2%}")
            
            # Update chart
            with chart_placeholder.container():
                if len(df_current) > 10:
                    fig = px.scatter(
                        df_current, x='packet_size', y='interval', 
                        color='is_botnet',
                        title="Real-time Network Traffic",
                        labels={'is_botnet': 'Traffic Type'},
                        color_discrete_map={0: 'green', 1: 'red'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Small delay for visualization
        time.sleep(0.5)
    
    status_placeholder.success("✅ Real-time simulation completed!")
    
    # Return final data
    return simulator._packets_to_dataframe()

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

# Handle simulation trigger
if hasattr(st.session_state, 'run_simulation') and st.session_state.run_simulation:
    if real_time_sim:
        # Run real-time simulation
        simulation_data = run_realtime_simulation()
        st.session_state.simulation_data = simulation_data
    else:
        # Run batch simulation
        with st.spinner("🔄 Running network simulation..."):
            simulation_data, stats = run_enhanced_simulation(
                output_file="simulated_robot_logs.csv",
                num_normal=num_normal,
                num_compromised=num_compromised,
                sim_time=sim_time
            )
        st.session_state.simulation_data = simulation_data
        
        # Display simulation statistics
        st.success("✅ Network simulation completed!")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Packets", f"{stats['total_packets']:,}")
        with col2:
            st.metric("Normal Traffic", f"{stats['normal_packets']:,}")
        with col3:
            st.metric("Botnet Traffic", f"{stats['botnet_packets']:,}")
        with col4:
            st.metric("Botnet Ratio", f"{stats['botnet_ratio']:.2%}")
    
    # Clear the trigger
    st.session_state.run_simulation = False
    st.rerun()

# Load data and train model
if hasattr(st.session_state, 'simulation_data') and st.session_state.simulation_data is not None:
    df = st.session_state.simulation_data
else:
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
    
    # Show simulation-specific information
    if 'timestamp' in df.columns:
        st.subheader("🌐 Network Simulation Details")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            duration = df['timestamp'].max() - df['timestamp'].min()
            st.metric("Simulation Duration", f"{duration:.1f}s")
        
        with col2:
            if 'source_id' in df.columns:
                unique_sources = df['source_id'].nunique()
                st.metric("Unique Sources", unique_sources)
        
        with col3:
            packets_per_sec = len(df) / duration if duration > 0 else 0
            st.metric("Packets/Second", f"{packets_per_sec:.1f}")
        
        with col4:
            avg_packet_size = df['packet_size'].mean()
            st.metric("Avg Packet Size", f"{avg_packet_size:.0f} bytes")
    
    # Dataset preview with enhanced columns
    st.subheader("🔍 Dataset Preview")
    display_df = df.head(10)
    if 'timestamp' in display_df.columns:
        display_df['timestamp'] = display_df['timestamp'].round(2)
    if 'interval' in display_df.columns:
        display_df['interval'] = display_df['interval'].round(3)
    st.dataframe(display_df)
    
    # Check if required columns exist
    required_columns = ['packet_size', 'interval']
    if all(col in df.columns for col in required_columns):
        
        # User input for prediction with interval-based analysis
        st.subheader("🔍 Botnet Detection & Analysis")
        
        # Create tabs for different prediction modes
        pred_tab1, pred_tab2 = st.tabs(["🎯 Single Prediction", "📊 Interval Analysis"])
        
        with pred_tab1:
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
        
        with pred_tab2:
            st.write("**Analyze Botnet Activity in Specific Ranges**")
            
            # Range selectors
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Packet Size Range**")
                packet_range = st.slider(
                    "Select packet size range (bytes)",
                    min_value=int(min_packet),
                    max_value=int(max_packet),
                    value=(int(min_packet), int(max_packet)),
                    step=50
                )
            
            with col2:
                st.write("**Interval Range**")
                interval_range = st.slider(
                    "Select interval range (seconds)",
                    min_value=float(min_interval),
                    max_value=float(max_interval),
                    value=(float(min_interval), float(max_interval)),
                    step=0.1,
                    format="%.2f"
                )
            
            if st.button("🔍 Analyze Range", type="primary", use_container_width=True):
                # Filter data based on ranges
                filtered_data = df[
                    (df['packet_size'] >= packet_range[0]) & 
                    (df['packet_size'] <= packet_range[1]) &
                    (df['interval'] >= interval_range[0]) & 
                    (df['interval'] <= interval_range[1])
                ]
                
                if len(filtered_data) > 0:
                    # Calculate statistics for the filtered range
                    total_packets = len(filtered_data)
                    botnet_packets = filtered_data['is_botnet'].sum()
                    normal_packets = total_packets - botnet_packets
                    botnet_ratio = botnet_packets / total_packets if total_packets > 0 else 0
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Packets", total_packets)
                    with col2:
                        st.metric("Normal Packets", int(normal_packets))
                    with col3:
                        st.metric("Botnet Packets", int(botnet_packets))
                    with col4:
                        color = "normal" if botnet_ratio < 0.3 else "inverse"
                        st.metric("Botnet Ratio", f"{botnet_ratio:.1%}", delta_color=color)
                    
                    # Risk assessment
                    if botnet_ratio > 0.7:
                        st.error("🚨 **HIGH RISK**: This range shows predominantly botnet activity!")
                    elif botnet_ratio > 0.3:
                        st.warning("⚠️ **MEDIUM RISK**: Elevated botnet activity in this range.")
                    else:
                        st.success("✅ **LOW RISK**: Range appears mostly normal.")
                    
                    # Visualization of the filtered data
                    fig = px.scatter(
                        filtered_data, x='packet_size', y='interval', color='is_botnet',
                        title=f"Activity in Selected Range (Packets: {packet_range[0]}-{packet_range[1]}, Intervals: {interval_range[0]:.2f}-{interval_range[1]:.2f})",
                        labels={'is_botnet': 'Traffic Type', 'packet_size': 'Packet Size (bytes)', 'interval': 'Interval (seconds)'},
                        color_discrete_map={0: 'green', 1: 'red'}
                    )
                    fig.update_traces(marker=dict(size=8, opacity=0.7))
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.warning("⚠️ No data found in the selected ranges. Try adjusting the ranges.")
        
        # Timeline visualization for simulation data (simplified)
        if 'timestamp' in df.columns and len(df) > 0:
            st.subheader("⏱️ Packet Size Over Time")
            
            # Create single timeline chart for packet sizes only
            normal_data = df[df['is_botnet'] == 0]
            botnet_data = df[df['is_botnet'] == 1]
            
            fig = go.Figure()
            
            if len(normal_data) > 0:
                fig.add_trace(
                    go.Scatter(x=normal_data['timestamp'], y=normal_data['packet_size'],
                              mode='markers', name='Normal Traffic',
                               marker=dict(color='green', size=4, opacity=0.6))
                )
            
            if len(botnet_data) > 0:
                fig.add_trace(
                    go.Scatter(x=botnet_data['timestamp'], y=botnet_data['packet_size'],
                              mode='markers', name='Botnet Traffic',
                               marker=dict(color='red', size=4, opacity=0.8))
                )
            
            fig.update_layout(
                height=400, 
                title="Packet Size Evolution During Simulation",
                xaxis_title="Time (seconds)",
                yaxis_title="Packet Size (bytes)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Visualization section
        st.subheader("📊 Data Visualization")
        
        if 'is_botnet' in df.columns and len(df) > 0:
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["🎯 Scatter Plot", "📈 Distributions", "⚖️ Class Balance", "🔬 Advanced Analysis"])
            
            with tab1:
                st.write("**Packet Size vs Interval Analysis**")
                
                # Use Plotly for interactive scatter plot
                fig = px.scatter(
                    df, x='packet_size', y='interval', color='is_botnet',
                    title="Network Traffic Patterns: Botnet vs Normal Activity",
                    labels={'is_botnet': 'Traffic Type', 'packet_size': 'Packet Size (bytes)', 'interval': 'Interval (seconds)'},
                    color_discrete_map={0: 'green', 1: 'red'},
                    hover_data=['timestamp'] if 'timestamp' in df.columns else None
                )
                fig.update_traces(marker=dict(size=6, opacity=0.7))
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interpretation
                st.info("""
                **Interpretation Guide:**
                - Normal: Typical network behavior
                - 🔴 **Red points (Botnet):** Potentially malicious activity
                - Look for clustering patterns that distinguish botnet from normal traffic
                - Interactive: Hover over points to see details, zoom and pan to explore
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
                    # Interactive pie chart
                    fig = px.pie(values=class_counts.values, names=['Normal', 'Botnet'],
                                title="Class Distribution",
                                color_discrete_map={'Normal': '#98FB98', 'Botnet': '#FFB6C1'})
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Interactive bar chart
                    fig = px.bar(x=['Normal', 'Botnet'], y=class_counts.values,
                                title="Activity Type Counts",
                                color=['Normal', 'Botnet'],
                                color_discrete_map={'Normal': '#98FB98', 'Botnet': '#FFB6C1'})
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Add statistics
                st.subheader("📈 Dataset Statistics")
                stats_col1, stats_col2 = st.columns(2)
                
                with stats_col1:
                    st.metric("Class Balance Ratio", f"{class_counts[0]/class_counts[1]:.2f}:1")
                    st.metric("Majority Class", f"{(class_counts.max()/len(df)*100):.1f}%")
                
                with stats_col2:
                    st.metric("Dataset Size", f"{len(df):,} samples")
                    st.metric("Feature Count", len(required_columns))
            
            with tab4:
                st.write("**Advanced Network Analysis**")
                
                # Packet size variation analysis
                if 'source_id' in df.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Packet Size Variations by Source")
                        
                        # Calculate standard deviation and mean for each source
                        source_stats = df.groupby('source_id').agg({
                            'packet_size': ['mean', 'std', 'count'],
                            'is_botnet': 'mean'
                        }).round(2)
                        
                        # Flatten column names
                        source_stats.columns = ['avg_packet_size', 'packet_size_std', 'packet_count', 'botnet_ratio']
                        source_stats = source_stats.fillna(0)
                        
                        # Get top sources by variation (std deviation)
                        top_variable = source_stats.nlargest(10, 'packet_size_std')
                        
                        if len(top_variable) > 0:
                            fig = px.bar(
                                x=top_variable.index,
                                y=top_variable['packet_size_std'],
                                title="Sources with Highest Packet Size Variation",
                                labels={'x': 'Source ID', 'y': 'Standard Deviation (bytes)'},
                                color=top_variable['botnet_ratio'],
                                color_continuous_scale='RdYlGn_r'
                            )
                            fig.update_xaxes(tickangle=45)
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Not enough source data for variation analysis")
                    
                    with col2:
                        st.subheader("⏱️ Interval Variations by Source")
                        
                        # Calculate interval variations
                        interval_stats = df.groupby('source_id').agg({
                            'interval': ['mean', 'std', 'count'],
                            'is_botnet': 'mean'
                        }).round(3)
                        
                        # Flatten column names
                        interval_stats.columns = ['avg_interval', 'interval_std', 'packet_count', 'botnet_ratio']
                        interval_stats = interval_stats.fillna(0)
                        
                        # Get top sources by interval variation
                        top_interval_var = interval_stats.nlargest(10, 'interval_std')
                        
                        if len(top_interval_var) > 0:
                            fig = px.bar(
                                x=top_interval_var.index,
                                y=top_interval_var['interval_std'],
                                title="Sources with Highest Interval Variation",
                                labels={'x': 'Source ID', 'y': 'Standard Deviation (seconds)'},
                                color=top_interval_var['botnet_ratio'],
                                color_continuous_scale='RdYlGn_r'
                            )
                            fig.update_xaxes(tickangle=45)
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Not enough source data for interval variation analysis")
                
                # Overall variation patterns
                st.subheader("📈 Traffic Pattern Variations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Packet size variance comparison
                    normal_var = df[df['is_botnet'] == 0]['packet_size'].var()
                    botnet_var = df[df['is_botnet'] == 1]['packet_size'].var()
                    
                    variance_data = pd.DataFrame({
                        'Traffic_Type': ['Normal', 'Botnet'],
                        'Packet_Size_Variance': [normal_var, botnet_var]
                    })
                    
                    fig = px.bar(
                        variance_data,
                        x='Traffic_Type',
                        y='Packet_Size_Variance',
                        title="Packet Size Variance: Normal vs Botnet",
                        color='Traffic_Type',
                        color_discrete_map={'Normal': 'green', 'Botnet': 'red'}
                    )
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Interval variance comparison
                    normal_interval_var = df[df['is_botnet'] == 0]['interval'].var()
                    botnet_interval_var = df[df['is_botnet'] == 1]['interval'].var()
                    
                    interval_variance_data = pd.DataFrame({
                        'Traffic_Type': ['Normal', 'Botnet'],
                        'Interval_Variance': [normal_interval_var, botnet_interval_var]
                    })
                    
                    fig = px.bar(
                        interval_variance_data,
                        x='Traffic_Type',
                        y='Interval_Variance',
                        title="Interval Variance: Normal vs Botnet",
                        color='Traffic_Type',
                        color_discrete_map={'Normal': 'green', 'Botnet': 'red'}
                    )
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # FIXED: Range-based analysis
                st.subheader("🎯 Packet Size & Interval Range Analysis")
                
                # Create range-based heatmap
                try:
                    # Create bins for packet size and interval with labels
                    packet_bins = pd.cut(df['packet_size'], bins=10, precision=0)
                    interval_bins = pd.cut(df['interval'], bins=10, precision=2)
                    
                    # Convert interval objects to string labels for JSON serialization
                    packet_labels = [str(interval) for interval in packet_bins.cat.categories]
                    interval_labels = [str(interval) for interval in interval_bins.cat.categories]
                    
                    # Create a copy of the dataframe with string bin labels
                    df_binned = df.copy()
                    df_binned['packet_bin'] = packet_bins.astype(str)
                    df_binned['interval_bin'] = interval_bins.astype(str)
                    
                    # Create pivot table for heatmap
                    heatmap_data = df_binned.groupby(['packet_bin', 'interval_bin'])['is_botnet'].agg(['count', 'mean']).fillna(0)
                    
                    if len(heatmap_data) > 0 and heatmap_data['count'].sum() > 0:
                        # Create heatmap of botnet ratio
                        pivot_botnet_ratio = heatmap_data['mean'].unstack(fill_value=0)
                        
                        # Only proceed if we have data
                        if not pivot_botnet_ratio.empty:
                            fig = px.imshow(
                                pivot_botnet_ratio.values,
                                x=pivot_botnet_ratio.columns,
                                y=pivot_botnet_ratio.index,
                                labels=dict(x="Interval Range", y="Packet Size Range", color="Botnet Ratio"),
                                title="Botnet Activity Heatmap by Packet Size and Interval Ranges",
                                color_continuous_scale="RdYlGn_r",
                                aspect="auto"
                            )
                            
                            # Improve layout
                            fig.update_layout(
                                height=500,
                                xaxis={'side': 'bottom'},
                                xaxis_tickangle=-45,
                                yaxis_tickangle=0
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show interpretation
                            st.info("""
                            **Heatmap Interpretation:**
                            - 🔴 **Red areas**: High botnet activity concentration
                            - 🟡 **Yellow areas**: Moderate botnet activity  
                            - 🟢 **Green areas**: Low/no botnet activity
                            - Use this to identify suspicious packet size and interval combinations
                            """)
                        else:
                            st.warning("Not enough varied data for range-based heatmap analysis")
                    else:
                        st.warning("Not enough data points for range-based heatmap analysis")
                    
                    # Alternative analysis: Simple range statistics
                    st.subheader("📊 Range-based Statistics Table")
                    
                    # Create quartile-based analysis as backup
                    packet_quartiles = df['packet_size'].quantile([0, 0.25, 0.5, 0.75, 1.0])
                    interval_quartiles = df['interval'].quantile([0, 0.25, 0.5, 0.75, 1.0])
                    
                    # Create range analysis table
                    range_analysis = []
                    
                    for i in range(4):  # 4 quartiles
                        packet_min = packet_quartiles.iloc[i]
                        packet_max = packet_quartiles.iloc[i + 1]
                        interval_min = interval_quartiles.iloc[i]
                        interval_max = interval_quartiles.iloc[i + 1]
                        
                        # Filter data for this quartile combination
                        quartile_data = df[
                            (df['packet_size'] >= packet_min) & (df['packet_size'] <= packet_max) &
                            (df['interval'] >= interval_min) & (df['interval'] <= interval_max)
                        ]
                        
                        if len(quartile_data) > 0:
                            botnet_ratio = quartile_data['is_botnet'].mean()
                            total_packets = len(quartile_data)
                            
                            range_analysis.append({
                                'Quartile': f'Q{i+1}',
                                'Packet Size Range': f'{packet_min:.0f} - {packet_max:.0f}',
                                'Interval Range': f'{interval_min:.3f} - {interval_max:.3f}',
                                'Total Packets': total_packets,
                                'Botnet Ratio': f'{botnet_ratio:.2%}',
                                'Risk Level': 'High' if botnet_ratio > 0.7 else 'Medium' if botnet_ratio > 0.3 else 'Low'
                            })
                    
                    if range_analysis:
                        range_df = pd.DataFrame(range_analysis)
                        
                        # Color-code the risk levels
                        def highlight_risk(val):
                            if val == 'High':
                                return 'background-color: #ffcccc'
                            elif val == 'Medium':
                                return 'background-color: #ffffcc'
                            else:
                                return 'background-color: #ccffcc'
                        
                        styled_df = range_df.style.applymap(highlight_risk, subset=['Risk Level'])
                        st.dataframe(styled_df, use_container_width=True)
                    
                    # Enhanced scatter plot with quartile boundaries
                    st.subheader("🎯 Enhanced Scatter Plot with Range Boundaries")
                    
                    fig = px.scatter(
                        df, x='packet_size', y='interval', color='is_botnet',
                        title="Network Traffic with Quartile Boundaries",
                        labels={'is_botnet': 'Traffic Type', 'packet_size': 'Packet Size (bytes)', 'interval': 'Interval (seconds)'},
                        color_discrete_map={0: 'green', 1: 'red'},
                        hover_data=['timestamp'] if 'timestamp' in df.columns else None
                    )
                    
                    # Add quartile lines
                    for q in packet_quartiles[1:-1]:  # Skip min and max
                        fig.add_vline(x=q, line_dash="dash", line_color="gray", opacity=0.5)
                    
                    for q in interval_quartiles[1:-1]:  # Skip min and max
                        fig.add_hline(y=q, line_dash="dash", line_color="gray", opacity=0.5)
                    
                    fig.update_traces(marker=dict(size=6, opacity=0.7))
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Could not create range analysis: {str(e)}")
                    st.info("Falling back to basic statistical analysis...")
                    
                    # Fallback: Simple statistical breakdown
                    st.subheader("📊 Basic Statistical Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Packet Size Analysis**")
                        normal_packets = df[df['is_botnet'] == 0]['packet_size']
                        botnet_packets = df[df['is_botnet'] == 1]['packet_size']
                        
                        packet_stats = pd.DataFrame({
                            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                            'Normal Traffic': [
                                normal_packets.mean(),
                                normal_packets.median(),
                                normal_packets.std(),
                                normal_packets.min(),
                                normal_packets.max()
                            ],
                            'Botnet Traffic': [
                                botnet_packets.mean(),
                                botnet_packets.median(),
                                botnet_packets.std(),
                                botnet_packets.min(),
                                botnet_packets.max()
                            ]
                        })
                        st.dataframe(packet_stats.round(2))
                    
                    with col2:
                        st.write("**Interval Analysis**")
                        normal_intervals = df[df['is_botnet'] == 0]['interval']
                        botnet_intervals = df[df['is_botnet'] == 1]['interval']
                        
                        interval_stats = pd.DataFrame({
                            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                            'Normal Traffic': [
                                normal_intervals.mean(),
                                normal_intervals.median(),
                                normal_intervals.std(),
                                normal_intervals.min(),
                                normal_intervals.max()
                            ],
                            'Botnet Traffic': [
                                botnet_intervals.mean(),
                                botnet_intervals.median(),
                                botnet_intervals.std(),
                                botnet_intervals.min(),
                                botnet_intervals.max()
                            ]
                        })
                        st.dataframe(interval_stats.round(3))
                
                # Statistical summary of variations
                st.subheader("📊 Variation Statistics Summary")
                
                variation_summary = pd.DataFrame({
                    'Metric': [
                        'Packet Size - Normal (Std Dev)',
                        'Packet Size - Botnet (Std Dev)', 
                        'Interval - Normal (Std Dev)',
                        'Interval - Botnet (Std Dev)',
                        'Packet Size - Normal (Range)',
                        'Packet Size - Botnet (Range)',
                        'Interval - Normal (Range)', 
                        'Interval - Botnet (Range)'
                    ],
                    'Value': [
                        df[df['is_botnet'] == 0]['packet_size'].std(),
                        df[df['is_botnet'] == 1]['packet_size'].std(),
                        df[df['is_botnet'] == 0]['interval'].std(),
                        df[df['is_botnet'] == 1]['interval'].std(),
                        df[df['is_botnet'] == 0]['packet_size'].max() - df[df['is_botnet'] == 0]['packet_size'].min(),
                        df[df['is_botnet'] == 1]['packet_size'].max() - df[df['is_botnet'] == 1]['packet_size'].min(),
                        df[df['is_botnet'] == 0]['interval'].max() - df[df['is_botnet'] == 0]['interval'].min(),
                        df[df['is_botnet'] == 1]['interval'].max() - df[df['is_botnet'] == 1]['interval'].min()
                    ]
                })
                
                variation_summary['Value'] = variation_summary['Value'].round(3)
                st.dataframe(variation_summary, use_container_width=True)
                
                # Temporal analysis (if timestamp available)
                if 'timestamp' in df.columns and len(df) > 0:
                    st.subheader("🕒 Temporal Pattern Analysis")
                    
                    try:
                        # Binned time analysis
                        df_temp = df.copy()
                        
                        # Ensure we have valid timestamp data
                        if df_temp['timestamp'].nunique() > 1:
                            df_temp['time_bin'] = pd.cut(df_temp['timestamp'], bins=min(20, df_temp['timestamp'].nunique()))
                            time_analysis = df_temp.groupby(['time_bin', 'is_botnet']).size().unstack(fill_value=0)
                            
                            if len(time_analysis) > 0:
                                # Create time bin labels for x-axis
                                time_labels = [f"Bin {i+1}" for i in range(len(time_analysis))]
                                
                                fig = go.Figure()
                                
                                # Add normal traffic trace
                                if 0 in time_analysis.columns:
                                    normal_data = time_analysis[0].values
                                    fig.add_trace(go.Scatter(
                                        x=time_labels,
                                        y=normal_data,
                                        mode='lines+markers',
                                        name='Normal Traffic',
                                        line=dict(color='green')
                                    ))
                                
                                # Add botnet traffic trace  
                                if 1 in time_analysis.columns:
                                    botnet_data = time_analysis[1].values
                                    fig.add_trace(go.Scatter(
                                        x=time_labels,
                                        y=botnet_data,
                                        mode='lines+markers',
                                        name='Botnet Traffic',
                                        line=dict(color='red')
                                    ))
                                
                                fig.update_layout(
                                    title="Traffic Patterns Over Time",
                                    xaxis_title="Time Bins",
                                    yaxis_title="Packet Count",
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Not enough data for temporal analysis")
                        else:
                            st.info("Insufficient timestamp variation for temporal analysis")
                    except Exception as e:
                        st.warning(f"Could not create temporal analysis: {str(e)}")
                        st.info("This feature requires diverse timestamp data from simulation")
                
                # Correlation analysis
                st.subheader("🔗 Feature Correlation Analysis")
                correlation_data = df[['packet_size', 'interval', 'is_botnet']].corr()
                
                fig = px.imshow(
                    correlation_data,
                    text_auto=True,
                    aspect="auto",
                    title="Feature Correlation Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("ℹ️ Visualization requires 'is_botnet' column in the dataset")
    
    else:
        st.error(f"❌ Required columns missing. Expected: {required_columns}, Found: {list(df.columns)}")

else:
    st.error("❌ Unable to load dataset or train model")

# Enhanced sidebar with simulation information
st.sidebar.header("ℹ️ About This Enhanced App")
st.sidebar.write("""
This application demonstrates **advanced botnet detection** with real network simulation capabilities using **SimPy**.

**New Features:**
- 🎮 Real-time network simulation
- 📡 SimPy-based traffic generation
- ⏱️ Timeline visualization
- 🤖 Multi-robot network modeling
- 🔄 Interactive simulation controls
""")

st.sidebar.header("🚀 How Simulation Works")
if SIMULATOR_AVAILABLE:
    st.sidebar.success("""
    **SimPy Integration Active:**
    1. **Network Setup:** Creates virtual robots and botnet controller
    2. **Traffic Generation:** Simulates realistic packet patterns
    3. **Real-time Updates:** Shows live network activity
    4. **Data Collection:** Captures timing and behavioral data
    5. **ML Training:** Uses simulated data for detection
    """)
else:
    st.sidebar.error("""
    **SimPy Not Available:**
    Install simpy to enable simulation features:
    ```bash
    pip install simpy
    ```
    """)

st.sidebar.header("📁 Enhanced File Support")
st.sidebar.write("""
**Simulation Data:**
- `simulated_robot_logs.csv` - SimPy generated data

**Legacy Support:**
- `synthetic_robot_logs.csv` - Custom dataset

**New Columns:**
- `timestamp` - Simulation time
- `source_id` - Robot identifier  
- `traffic_type` - Traffic classification
- `destination_id` - Target endpoint
""")

st.sidebar.header("🔧 Technical Stack")
tech_status = []
tech_status.append("✅ Streamlit - Web Interface")
tech_status.append("✅ scikit-learn - ML Models")
tech_status.append("✅ Plotly - Interactive Charts")
tech_status.append("✅ Pandas - Data Processing")

if SIMULATOR_AVAILABLE:
    tech_status.append("✅ SimPy - Network Simulation")
else:
    tech_status.append("❌ SimPy - Not Available")

for status in tech_status:
    st.sidebar.write(status)

# Performance metrics section
if model is not None and df is not None:
    with st.expander("📊 Enhanced Model Performance"):
        st.subheader("Model Configuration")
        st.code(f"""
Random Forest Classifier:
- Estimators: 100
- Max Depth: 10
- Min Samples Split: 5
- Random State: 42
- Test Accuracy: {model_accuracy:.3f}
- Dataset Source: {'Simulation' if 'timestamp' in df.columns else 'Static'}
        """)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': required_columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                importance_df,
                x='Feature',
                y='Importance',
                title="Feature Importance in Botnet Detection"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Model performance over time (if timestamp available)
        if 'timestamp' in df.columns and len(df) > 100:
            st.subheader("Model Performance Over Time")
            
            try:
                # Split data into time windows and evaluate
                df_sorted = df.sort_values('timestamp')
                window_size = max(50, len(df_sorted) // 10)  # Ensure minimum window size
                
                performance_over_time = []
                for i in range(0, len(df_sorted) - window_size, max(1, window_size // 2)):
                    window_data = df_sorted.iloc[i:i+window_size]
                    if len(window_data) > 10 and window_data['is_botnet'].nunique() > 1:
                        try:
                            X_window = window_data[['packet_size', 'interval']].values
                            y_window = window_data['is_botnet'].values
                            
                            # Check for valid data
                            if len(X_window) > 0 and len(y_window) > 0:
                                X_window_scaled = scaler.transform(X_window)
                                y_pred_window = model.predict(X_window_scaled)
                                
                                accuracy = accuracy_score(y_window, y_pred_window)
                                avg_timestamp = window_data['timestamp'].mean()
                                
                                performance_over_time.append({
                                    'timestamp': avg_timestamp,
                                    'accuracy': accuracy
                                })
                        except Exception as window_error:
                            continue  # Skip problematic windows
                
                if performance_over_time:
                    perf_df = pd.DataFrame(performance_over_time)
                    fig = px.line(
                        perf_df,
                        x='timestamp',
                        y='accuracy',
                        title="Model Accuracy Over Simulation Time",
                        labels={'timestamp': 'Simulation Time (s)', 'accuracy': 'Accuracy'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data windows for performance analysis over time")
            except Exception as e:
                st.warning(f"Could not analyze performance over time: {str(e)}")
                st.info("This feature requires substantial timestamp data from simulation")
