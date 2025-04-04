#!/usr/bin/env python3
import rosbag
import pandas as pd
import os
import numpy as np
from collections import defaultdict

def bag_to_csv(bag_names):
    for bag_name in bag_names:
        bag_name = '/home/szylzz/Desktop/aps360-project/data/bag/'+ bag_name
        bag_file = bag_name + '.bag'
        
        # Output directory for CSV files (same directory as the script)
        output_dir = os.path.dirname(os.path.abspath(__file__))

        # Read the ROS bag file
        print(f"Opening bag file: {bag_file}")
        bag = rosbag.Bag(bag_file)

        # Get all topics in the bag
        topic_info = bag.get_type_and_topic_info()[1]
        all_topics = list(topic_info.keys())
        print(f"Found {len(all_topics)} topics in the bag file:")
        for topic in all_topics:
            print(f"  - {topic} ({topic_info[topic].msg_type})")

        # Dictionary to store data for each topic
        topic_data = defaultdict(list)

        # Process all messages
        print("\nProcessing messages...")
        message_count = 0

        for topic, msg, t in bag.read_messages():
            # Create a dictionary for this message
            row = {"timestamp": t.to_sec()}
            
            # Extract message fields
            for field in msg.__slots__:
                value = getattr(msg, field)
                
                # Handle nested messages
                if hasattr(value, '__slots__'):
                    for sub_field in value.__slots__:
                        sub_value = getattr(value, sub_field)
                        
                        # Handle nested nested messages (e.g., for Pose, Twist, etc.)
                        if hasattr(sub_value, '__slots__'):
                            for sub_sub_field in sub_value.__slots__:
                                row[f"{field}.{sub_field}.{sub_sub_field}"] = getattr(sub_value, sub_sub_field)
                        else:
                            row[f"{field}.{sub_field}"] = sub_value
                else:
                    row[field] = value
            
            # Add to the appropriate topic data
            topic_data[topic].append(row)
            message_count += 1
            
            # Print progress every 10000 messages
            if message_count % 10000 == 0:
                print(f"Processed {message_count} messages...")

        bag.close()
        print(f"Finished processing {message_count} messages")

        # Convert data for each topic to a CSV file
        print("\nSaving CSVs...")
        for topic in topic_data:
            # Create a safe filename from the topic name
            safe_topic = topic.replace('/', '_').strip('_')
            output_csv = os.path.join(output_dir, f"{bag_name}_{safe_topic}.csv")
            
            # Convert to DataFrame
            df = pd.DataFrame(topic_data[topic])
            
            # Clean up any potential issues for CSV output
            for col in df.columns:
                # Convert numpy arrays or lists to strings
                if df[col].dtype == 'object':
                    df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (np.ndarray, list)) else x)
            
            # Save to CSV
            df.to_csv(output_csv, index=False)
            print(f"Saved {len(topic_data[topic])} messages from topic '{topic}' to {output_csv}")

        print("\nConversion completed!")

# Input ROS bag file
bag_names = [
    '2025-04-03-19-54-22',
    '2025-04-03-19-55-56',
    '2025-04-03-19-56-41',
    '2025-04-03-19-57-11',
    '2025-04-03-19-57-44',
    '2025-04-03-19-58-12',
    '2025-04-03-19-58-53',
    '2025-04-03-19-59-26',
    '2025-04-03-20-00-38',
    '2025-04-03-20-01-06',
    '2025-03-07-23-28-38']

bag_to_csv(bag_names)