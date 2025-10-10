# Test for synchronization

## 1. Run the server script on your master machine:
```bash
python server.py
```

## 2. Run the node script on a compute node in the cluster:
```bash
python node.py <n>
```
where `<n>` can be any value from 1 to 5.

> **Note:** The server and node scripts must be run on machines within the same local area network (LAN) to ensure they can communicate with each other.
