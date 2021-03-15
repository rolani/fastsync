import time
import ray
import random
import Server1
import sys

num_workers = int(sys.argv[1])

###########################################################################
# Fast Synchronous Parameter Server Training
# --------------------------------------
# We'll now create a synchronous parameter server training scheme. We'll first
# instantiate a process for the parameter server, along with multiple
# workers.

iterations = 100
#num_workers = 5
quorum_num = round(0.7 * num_workers)
out_file = "Fast_Sync_out{0}_acc_loss.txt".format(num_workers)
time_file = "Fast_Sync_out{0}_iter_time.txt".format(num_workers)

f_out = open(out_file, "a")
f_time = open(time_file, "a")

print("Running Fast Sync Parameter Server Training.")

ray.init(ignore_reinit_error=True)
ps = Server1.ParameterServer.remote(1e-2)
workers = [Server1.Worker.remote() for i in range(num_workers)]

###########################################################################
# Here, workers will asynchronously compute the gradients given its
# current weights and send these gradients to the parameter server as
# soon as they are ready. When the Parameter server finishes applying the
# new gradient, the server will send back a copy of the current weights to the
# worker. The worker will then update the weights and repeat.

###########################################################################
# We'll also instantiate a model on the driver process to evaluate the test
# accuracy during training.

model = Server1.ConvNet()
test_loader = Server1.get_data_loader()[1]

###########################################################################
start_time = time.time()
init_start_time = time.time()
current_weights = ps.get_weights.remote()

#initial computation
gradients = {}
for worker in workers:
    gradients[worker.compute_gradients.remote(current_weights)] = worker
print("-----gradients-----")
print(gradients)

# main training iteration
accuracy = 0
i = 0
while accuracy <= 60:
    print("Iteration {0}".format(i))
    worker_list_to_run = []
    ready_worker_list,_ = ray.wait(list(gradients), num_returns=quorum_num)

    for val_ in ready_worker_list:
        worker_ = gradients[val_]
        worker_list_to_run.append(worker_)
        gradients.pop(val_)

    current_weights = ps.apply_gradients.remote(*ready_worker_list)

    for worker in worker_list_to_run:
        gradients[worker.compute_gradients.remote(current_weights)] = worker

    if i % 2000 == 0:
        # Evaluate the current model.
        f_time.write("{} {:.5f}".format(i, time.time() - start_time))
        model.set_weights(ray.get(current_weights))
        loss, accuracy = Server1.evaluate(model, test_loader)
        print("Iter {}: \taccuracy is {:.3f} \tloss is {:.6f}".format(i, accuracy, loss))
        # end_time = time.time()
        f_out.write('{} {:3f} {:6f}\n'.format(i, accuracy, loss))
        start_time = time.time()
        f_time.write(" {:.3f}\n".format(accuracy))
    i += 1

#model.set_weights(ray.get(current_weights))
#loss, accuracy = Server.evaluate(model, test_loader)
#print("Iter {}: \taccuracy is {:.3f} \tloss is {:.6f}".format(i, accuracy, loss))
end_time = time.time()
f_time.write("{:.5f}".format(time.time() - init_start_time))
print("Final Runtime is: {0}".format(time.time() - init_start_time))
#f_time.write("{:.5f}".format(time.time() - start_time))
print("Final Runtime is: {0}".format(time.time() - start_time))
print("Final accuracy is {:.3f}.".format(accuracy))
print("Final loss is {:.6f}.".format(loss))
# Clean up Ray resources and processes before the next example.
ray.shutdown()
print("Fast sync training completed with {0} workers".format(num_workers))
f_time.close()
f_out.close()

