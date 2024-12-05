import streamlit as st

def display_company_info(stock):
    """Display company information and profile using responsive containers"""
    info = stock.info
    
    # Main container for company profile
    with st.container():
        st.subheader("Company Profile")
        
        # Use columns for basic info
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div style='background-color: {st.get_option('theme.secondaryBackgroundColor')}; 
                           padding: 1rem; border-radius: 0.5rem;'>
                    <h4>Sector</h4>
                    {info.get('sector', 'N/A')}
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""
                <div style='background-color: {st.get_option('theme.secondaryBackgroundColor')}; 
                           padding: 1rem; border-radius: 0.5rem;'>
                    <h4>Industry</h4>
                    {info.get('industry', 'N/A')}
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Business Summary section with expander
    with st.container():
        st.markdown("### Business Summary")
        business_summary = info.get('longBusinessSummary', 'No information available')
        
        # Create a unique key for this instance
        state_key = f"show_full_{info.get('symbol', 'unknown')}"
        
        # Initialize session state for the toggle if it doesn't exist
        if state_key not in st.session_state:
            st.session_state[state_key] = False

        # Create a container with custom styling for the summary
        summary_container = st.container()
        
        # Define the button callbacks
        def show_more():
            st.session_state[state_key] = True
            
        def show_less():
            st.session_state[state_key] = False

        # Display the content in the styled container
        with summary_container:
            st.markdown(
                f"""
                <div style='background-color: {st.get_option('theme.secondaryBackgroundColor')}; 
                           padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0;'>
                    {business_summary[:300] + "..." if len(business_summary) > 300 and not st.session_state[state_key] else business_summary}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            if len(business_summary) > 300:
                if not st.session_state[state_key]:
                    st.button("Show More", 
                             key=f"more_{info.get('symbol', 'unknown')}", 
                             on_click=show_more)
                else:
                    st.button("Show Less", 
                             key=f"less_{info.get('symbol', 'unknown')}", 
                             on_click=show_less)
    
    # Key Statistics section using columns for responsive layout
    with st.container():
        st.markdown("### Key Statistics")
        
        # Create stats dictionary
        stats = {
            'Beta': info.get('beta'),
            '52 Week High': info.get('fiftyTwoWeekHigh'),
            '52 Week Low': info.get('fiftyTwoWeekLow'),
            'Volume': info.get('volume'),
            'Avg Volume': info.get('averageVolume')
        }
        
        # Display stats in a responsive grid
        cols = st.columns(3)
        for idx, (key, value) in enumerate(stats.items()):
            with cols[idx % 3]:
                st.markdown(
                    f"""
                    <div style='background-color: {st.get_option('theme.secondaryBackgroundColor')}; 
                               padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;'>
                        <h4>{key}</h4>
                        {value if value is not None else 'N/A'}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
