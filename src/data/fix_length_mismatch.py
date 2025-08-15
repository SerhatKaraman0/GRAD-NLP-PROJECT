import pandas as pd
import logging

logger = logging.getLogger(__name__)

def fix_dataframe_length_mismatch(original_df, processed_df):
    """
    Fix length mismatch issues by ensuring processed_df has the same length as original_df,
    properly handling padding and truncation for different data types.
    
    Args:
        original_df (pd.DataFrame): The original dataframe
        processed_df (pd.DataFrame): The processed dataframe (may have different length)
        
    Returns:
        pd.DataFrame: A fixed dataframe with the same length as the original
    """
    original_length = len(original_df)
    processed_length = len(processed_df)
    
    if original_length == processed_length:
        # No mismatch, return as is
        return processed_df
    
    logger.warning(f"Length mismatch detected: original={original_length}, processed={processed_length}")
    
    # Create a DataFrame with the same index as the original
    fixed_df = pd.DataFrame(index=range(original_length))
    
    # Identify columns to preserve from the original dataframe
    # These are typically metadata columns like IDs, ratings, etc.
    preserve_columns = ['Id', 'Score', 'Summary', 'ProductId', 'UserId', 'Text_hash']
    preserve_columns = [col for col in preserve_columns if col in original_df.columns]
    
    # Add preserve columns first if they exist
    if preserve_columns:
        for col in preserve_columns:
            if col in original_df.columns:
                fixed_df[col] = original_df[col].values
    
    # Add processed data columns with proper padding/truncation
    for col in processed_df.columns:
        if col in preserve_columns:
            continue  # Already added
            
        # Get the data to assign
        data = processed_df[col].values
        
        # Handle length mismatch
        if len(data) < original_length:
            # Need to pad with default values based on column type
            if col.startswith(('Tokens', 'Sentences', 'Words')):
                # List-type data
                padding = [[] for _ in range(original_length - len(data))]
            elif col.startswith(('Entities', 'Tags', 'Metadata')):
                # Dict-type data
                padding = [{} for _ in range(original_length - len(data))]
            elif col.startswith(('Quality', 'Score', 'Sentiment', 'Subjectivity', 'Confidence')):
                # Float-type data
                padding = [0.0 for _ in range(original_length - len(data))]
            elif col.startswith(('Count', 'Length', 'Num')):
                # Int-type data
                padding = [0 for _ in range(original_length - len(data))]
            else:
                # Default to empty string for string-type data
                padding = [''] * (original_length - len(data))
                
            # Extend the data with padding
            data = list(data) + padding
        elif len(data) > original_length:
            # Need to truncate
            data = data[:original_length]
        
        # Assign to the fixed dataframe
        fixed_df[col] = data
    
    logger.info("Length mismatch fixed successfully")
    return fixed_df
