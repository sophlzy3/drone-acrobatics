#!/usr/bin/env python3
import rosbag
import pandas as pd
import os,sys
import numpy as np
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.vars import BAGS_BASELINE, BAGS_TRAINING

import os
import pandas as pd
import numpy as np
from collections import defaultdict
import bagpy
from bagpy import bagreader

def bag_to_csv(path, bag_names):
    for bag_name in bag_names:
        bag_path = path + bag_name + '.bag'
        
        print(f"Reading bag file: {bag_path}")
        b = bagreader(bag_path)

        all_topics = b.topic_table['Topics'].tolist()
        print(f"Found {len(all_topics)} topics:")
        for topic in all_topics:
            print(f"  - {topic}")
        
        for topic in all_topics:
            print(f"Processing topic: {topic}")
            try:
                csv_filename = b.message_by_topic(topic)  # returns path to csv
                df = pd.read_csv(csv_filename)

                safe_topic = topic.replace('/', '_').strip('_')
                final_csv_path = path + f"{bag_name}_{safe_topic}.csv"
                
                df.to_csv(final_csv_path, index=False)
                print(f"Saved topic '{topic}' to {final_csv_path}")
            except Exception as e:
                print(f"Failed to process topic {topic}: {e}")

        print("\nFinished converting bag to CSV!\n")

bag_path = '/Users/sophie/Downloads/Github/aps360-project/data/baseline_bag/'
bag_to_csv(bag_path,BAGS_BASELINE)