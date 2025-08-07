import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from typing import Dict, Any, List
import json

def admin_page():
    """Admin page for query optimization metrics and performance monitoring"""
    
    # Note: page_config should be set in the calling page, not here
    
    # Header with navigation
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🚀 ABI Admin Dashboard")
        st.markdown("### Query Performance & Optimization Metrics")
    with col2:
        if st.button("🏠 Back to Main"):
            st.switch_page("main.py")
    
    # Check if app is initialized
    if 'app' not in st.session_state:
        st.error("❌ No data loaded. Please go to the main page and upload some data first.")
        st.info("Navigate to the main page to upload CSV, Excel, or text files.")
        return
    
    app = st.session_state.app
    
    # Sidebar for admin controls
    with st.sidebar:
        st.header("🔧 Admin Controls")
        
        # Cache management
        st.subheader("Cache Management")
        if st.button("🗑️ Clear Query Cache"):
            app.query_optimizer.query_cache.clear()
            st.success("Query cache cleared!")
        
        if st.button("🗑️ Clear Performance Metrics"):
            app.performance_metrics.clear()
            app.query_history.clear()
            st.success("Performance metrics cleared!")
        
        # Export data
        st.subheader("Export Data")
        if st.button("📊 Export Performance Data"):
            export_performance_data(app)
        
        # System info
        st.subheader("System Information")
        st.write(f"**Datasets Loaded:** {len(app.dfs)}")
        st.write(f"**Text Files:** {len(app.text_files)}")
        st.write(f"**Total Queries:** {len(app.performance_metrics)}")
        
        if app.vectorstore:
            st.success("✅ Vector store initialized")
        else:
            st.error("❌ Vector store not initialized")
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Performance Overview", 
        "🔍 Query Analysis", 
        "⚡ Strategy Performance", 
        "💡 Optimization Recommendations",
        "📊 Detailed Metrics"
    ])
    
    with tab1:
        display_performance_overview(app)
    
    with tab2:
        display_query_analysis(app)
    
    with tab3:
        display_strategy_performance(app)
    
    with tab4:
        display_optimization_recommendations(app)
    
    with tab5:
        display_detailed_metrics(app)

def display_performance_overview(app):
    """Display high-level performance metrics"""
    st.header("📈 Performance Overview")
    
    insights = app.get_performance_insights()
    
    if 'message' in insights:
        st.info(insights['message'])
        return
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Queries", 
            insights['total_queries'],
            help="Total number of queries executed"
        )
    
    with col2:
        cache_rate = insights['cache_hit_rate']
        st.metric(
            "Cache Hit Rate", 
            f"{cache_rate:.1%}",
            delta=f"{cache_rate:.1%}" if cache_rate > 0.2 else f"-{cache_rate:.1%}",
            delta_color="normal" if cache_rate > 0.2 else "inverse",
            help="Percentage of queries served from cache"
        )
    
    with col3:
        avg_time = insights['average_execution_time']
        st.metric(
            "Avg Execution Time", 
            f"{avg_time:.2f}s",
            delta=f"{avg_time:.2f}s",
            delta_color="inverse" if avg_time < 3.0 else "normal",
            help="Average time to process queries"
        )
    
    with col4:
        if app.query_history:
            recent_queries = [q for q in app.query_history if time.time() - q['timestamp'] < 3600]  # Last hour
            st.metric(
                "Queries (Last Hour)", 
                len(recent_queries),
                help="Number of queries in the last hour"
            )
        else:
            st.metric("Queries (Last Hour)", 0)
    
    # Performance trends chart
    if app.query_history:
        st.subheader("📊 Query Performance Trends")
        
        # Create time series data
        df_history = pd.DataFrame(app.query_history)
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp'], unit='s')
        df_history['hour'] = df_history['timestamp'].dt.floor('H')
        
        # Group by hour
        hourly_stats = df_history.groupby('hour').agg({
            'execution_time': ['mean', 'count'],
            'complexity': lambda x: x.value_counts().index[0] if len(x) > 0 else 'low'
        }).reset_index()
        
        hourly_stats.columns = ['hour', 'avg_time', 'query_count', 'most_common_complexity']
        
        # Create trend chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=hourly_stats['hour'],
            y=hourly_stats['avg_time'],
            mode='lines+markers',
            name='Avg Execution Time (s)',
            yaxis='y'
        ))
        
        fig.add_trace(go.Bar(
            x=hourly_stats['hour'],
            y=hourly_stats['query_count'],
            name='Query Count',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Query Performance Over Time",
            xaxis_title="Time",
            yaxis=dict(title="Execution Time (s)", side="left"),
            yaxis2=dict(title="Query Count", side="right", overlaying="y"),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_query_analysis(app):
    """Display detailed query analysis"""
    st.header("🔍 Query Analysis")
    
    if not app.query_history:
        st.info("No query history available. Run some queries to see analysis.")
        return
    
    # Query complexity distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Query Complexity Distribution")
        complexity_counts = {}
        for query in app.query_history:
            complexity = query['complexity']
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        if complexity_counts:
            fig = px.pie(
                values=list(complexity_counts.values()),
                names=list(complexity_counts.keys()),
                title="Query Complexity Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Query Strategy Usage")
        strategy_counts = {}
        for query in app.query_history:
            strategy = query['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        if strategy_counts:
            fig = px.bar(
                x=list(strategy_counts.keys()),
                y=list(strategy_counts.values()),
                title="Search Strategy Usage"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent queries table
    st.subheader("📝 Recent Queries")
    if app.query_history:
        recent_queries = sorted(app.query_history, key=lambda x: x['timestamp'], reverse=True)[:20]
        
        query_data = []
        for query in recent_queries:
            query_data.append({
                'Query': query['query'][:50] + "..." if len(query['query']) > 50 else query['query'],
                'Strategy': query['strategy'],
                'Complexity': query['complexity'],
                'Execution Time (s)': f"{query['execution_time']:.2f}",
                'Timestamp': datetime.fromtimestamp(query['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            })
        
        df_queries = pd.DataFrame(query_data)
        st.dataframe(df_queries, use_container_width=True)

def display_strategy_performance(app):
    """Display detailed strategy performance analysis"""
    st.header("⚡ Strategy Performance")
    
    insights = app.get_performance_insights()
    
    if 'message' in insights or not insights['strategy_performance']:
        st.info("No strategy performance data available. Run some queries to see analysis.")
        return
    
    # Strategy comparison table
    st.subheader("📊 Strategy Performance Comparison")
    
    strategy_data = []
    for strategy, perf in insights['strategy_performance'].items():
        strategy_data.append({
            'Strategy': strategy,
            'Query Count': perf['count'],
            'Avg Time (s)': f"{perf['average_time']:.2f}",
            'Min Time (s)': f"{perf['min_time']:.2f}",
            'Max Time (s)': f"{perf['max_time']:.2f}",
            'Efficiency Score': f"{perf['count'] / perf['average_time']:.1f}"
        })
    
    strategy_df = pd.DataFrame(strategy_data)
    strategy_df = strategy_df.sort_values('Efficiency Score', ascending=False)
    st.dataframe(strategy_df, use_container_width=True)
    
    # Strategy performance chart
    st.subheader("📈 Strategy Performance Visualization")
    
    fig = go.Figure()
    
    strategies = list(insights['strategy_performance'].keys())
    avg_times = [insights['strategy_performance'][s]['average_time'] for s in strategies]
    counts = [insights['strategy_performance'][s]['count'] for s in strategies]
    
    fig.add_trace(go.Bar(
        x=strategies,
        y=avg_times,
        name='Average Time (s)',
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=strategies,
        y=counts,
        name='Query Count',
        yaxis='y2',
        mode='lines+markers'
    ))
    
    fig.update_layout(
        title="Strategy Performance Comparison",
        xaxis_title="Strategy",
        yaxis=dict(title="Average Time (s)", side="left"),
        yaxis2=dict(title="Query Count", side="right", overlaying="y"),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_optimization_recommendations(app):
    """Display optimization recommendations"""
    st.header("💡 Optimization Recommendations")
    
    recommendations = app.get_query_optimization_recommendations()
    
    if not recommendations:
        st.success("✅ No optimization recommendations at this time.")
        return
    
    # Display recommendations with icons
    for i, rec in enumerate(recommendations, 1):
        if "✅" in rec:
            st.success(f"{i}. {rec}")
        elif "🔧" in rec or "⚡" in rec:
            st.warning(f"{i}. {rec}")
        elif "⏱️" in rec:
            st.error(f"{i}. {rec}")
        else:
            st.info(f"{i}. {rec}")
    
    # Performance improvement suggestions
    st.subheader("🚀 Performance Improvement Suggestions")
    
    insights = app.get_performance_insights()
    if 'message' not in insights:
        suggestions = []
        
        # Cache optimization suggestions
        if insights['cache_hit_rate'] < 0.1:
            suggestions.append("**Low Cache Hit Rate**: Consider implementing query similarity caching or increasing cache TTL")
        
        # Strategy optimization suggestions
        if insights['strategy_performance']:
            slowest_strategy = min(insights['strategy_performance'].items(), 
                                 key=lambda x: x[1]['average_time'])
            if slowest_strategy[1]['average_time'] > 5.0:
                suggestions.append(f"**Slow Strategy**: {slowest_strategy[0]} is taking {slowest_strategy[1]['average_time']:.2f}s on average. Consider optimization.")
        
        # Query complexity suggestions
        if insights['complexity_distribution'].get('high', 0) > insights['complexity_distribution'].get('low', 0):
            suggestions.append("**High Complexity Queries**: Many complex queries detected. Consider query simplification or better preprocessing.")
        
        if suggestions:
            for suggestion in suggestions:
                st.info(suggestion)
        else:
            st.success("✅ Performance looks good! No immediate improvements needed.")

def display_detailed_metrics(app):
    """Display detailed performance metrics"""
    st.header("📊 Detailed Performance Metrics")
    
    insights = app.get_performance_insights()
    
    if 'message' in insights:
        st.info(insights['message'])
        return
    
    # Detailed metrics in expandable sections
    with st.expander("🔍 Cache Performance Details", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Cache Hit Rate", f"{insights['cache_hit_rate']:.1%}")
            st.metric("Cache Miss Rate", f"{1 - insights['cache_hit_rate']:.1%}")
        
        with col2:
            if app.query_optimizer.query_cache:
                cache_size = len(app.query_optimizer.query_cache)
                st.metric("Cache Size", cache_size)
                
                # Calculate cache age
                if cache_size > 0:
                    oldest_entry = min(app.query_optimizer.query_cache.values(), 
                                     key=lambda x: x['timestamp'])
                    cache_age = time.time() - oldest_entry['timestamp']
                    st.metric("Oldest Cache Entry", f"{cache_age/60:.1f} minutes")
    
    with st.expander("⚡ Execution Time Breakdown", expanded=True):
        if app.performance_metrics:
            # Calculate average times for each component
            component_times = {
                'preprocessing_time': [],
                'search_time': [],
                'processing_time': [],
                'context_time': [],
                'response_time': []
            }
            
            for query_hash, metrics in app.performance_metrics.items():
                for metric in metrics:
                    if 'type' in metric and metric['type'] == 'full_query':
                        for component in component_times:
                            if component in metric:
                                component_times[component].append(metric[component])
            
            # Display averages
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if component_times['preprocessing_time']:
                    avg_prep = sum(component_times['preprocessing_time']) / len(component_times['preprocessing_time'])
                    st.metric("Avg Preprocessing", f"{avg_prep:.3f}s")
                
                if component_times['search_time']:
                    avg_search = sum(component_times['search_time']) / len(component_times['search_time'])
                    st.metric("Avg Search", f"{avg_search:.3f}s")
            
            with col2:
                if component_times['processing_time']:
                    avg_proc = sum(component_times['processing_time']) / len(component_times['processing_time'])
                    st.metric("Avg Processing", f"{avg_proc:.3f}s")
                
                if component_times['context_time']:
                    avg_context = sum(component_times['context_time']) / len(component_times['context_time'])
                    st.metric("Avg Context Building", f"{avg_context:.3f}s")
            
            with col3:
                if component_times['response_time']:
                    avg_response = sum(component_times['response_time']) / len(component_times['response_time'])
                    st.metric("Avg Response Generation", f"{avg_response:.3f}s")
    
    with st.expander("🐌 Slowest Queries", expanded=True):
        if insights['slowest_queries']:
            slow_queries_data = []
            for query_hash, execution_time in insights['slowest_queries']:
                slow_queries_data.append({
                    'Query Hash': query_hash[:8] + "...",
                    'Execution Time (s)': f"{execution_time:.2f}",
                    'Full Hash': query_hash
                })
            
            slow_df = pd.DataFrame(slow_queries_data)
            st.dataframe(slow_df, use_container_width=True)
        else:
            st.info("No slow queries detected.")

def export_performance_data(app):
    """Export performance data for analysis"""
    try:
        # Prepare data for export
        export_data = {
            'performance_metrics': app.performance_metrics,
            'query_history': app.query_history,
            'export_timestamp': datetime.now().isoformat()
        }
        
        # Convert to JSON
        json_data = json.dumps(export_data, indent=2, default=str)
        
        # Create download button
        st.download_button(
            label="📥 Download Performance Data (JSON)",
            data=json_data,
            file_name=f"abi_performance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        st.success("✅ Performance data ready for download!")
        
    except Exception as e:
        st.error(f"❌ Error exporting data: {str(e)}")

if __name__ == "__main__":
    admin_page() 