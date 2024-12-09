import wfdb
import numpy as np

class WfdbReader:
    data_path = './data/mit-bih-noise-stress-test-database-1.0.0/'
    record_names = [
        "118e24", "119e24", "118e18", "119e18", "118e12", "119e12",
        "118e06", "119e06", "118e00", "119e00", "118e_6", "119e_6"
    ]

    def fetch_data(self):
        data_list = []
        for record_name in self.record_names:
            record = wfdb.rdrecord(self.data_path + record_name)
            data_list.append(record.p_signal)
        return np.stack(data_list)


