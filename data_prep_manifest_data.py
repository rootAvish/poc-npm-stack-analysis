import numpy as np
pairs_train = []
pairs_test = []
num_train_per_user = 5
user_id = 0
np.random.seed(123)
for line in open("manifest_user_data.dat"):
    arr = line.strip().split()
    arr = np.asarray([int(x) for x in arr[1:]])
    n = len(arr)
    idx = np.random.permutation(n)
    # assert(n > num_train_per_user)
    for i in range(min(num_train_per_user, n)):
        # Add num_train_per_user or all of the user's items to the training data.
        pairs_train.append((user_id, arr[idx[i]]))
    # if we have more items than we need for training, append to testing.
    if n > num_train_per_user:
        for i in range(num_train_per_user, n):
            pairs_test.append((user_id, arr[idx[i]]))
    user_id += 1
num_users = user_id
pairs_train = np.asarray(pairs_train)
pairs_test = np.asarray(pairs_test)
num_items = np.maximum(np.max(pairs_train[:, 1]), np.max(pairs_test[:, 1]))+1
print("num_users=%d, num_items=%d" % (num_users, num_items))

# Row - users, column - items
with open("packagedata-train-"+str(num_train_per_user)+"-users.dat", "w") as fid:
    print(fid.name)
    for user_id in range(num_users):
        # Collect all items of this user.
        this_user_items = pairs_train[pairs_train[:, 0]==user_id, 1]
        # Convert to a space separated string of integers
        items_str = " ".join(str(x) for x in this_user_items)
        fid.write("%d %s\n" % (len(this_user_items), items_str))

# Row - items, column - users
with open("packagedata-train-"+str(num_train_per_user)+"-items.dat", "w") as fid:
    print(fid.name)
    for item_id in range(num_items):
        this_item_users = pairs_train[pairs_train[:, 1]==item_id, 0]
        users_str = " ".join(str(x) for x in this_item_users)
        fid.write("%d %s\n" % (len(this_item_users), users_str))

with open("packagedata-test-"+str(num_train_per_user)+"-users.dat", "w") as fid:
    for user_id in range(num_users):
        this_user_items = pairs_test[pairs_test[:, 0]==user_id, 1]
        items_str = " ".join(str(x) for x in this_user_items)
        fid.write("%d %s\n" % (len(this_user_items), items_str))

with open("packagedata-test-"+str(num_train_per_user)+"-items.dat", "w") as fid:
    print(fid.name)
    for item_id in range(num_items):
        this_item_users = pairs_test[pairs_test[:, 1]==item_id, 0]
        users_str = " ".join(str(x) for x in this_item_users)
        fid.write("%d %s\n" % (len(this_item_users), users_str))
