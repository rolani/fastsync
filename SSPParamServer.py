import time
import ray
import random
import Server1
import sys

###########################################################################
# Stale Synchronous Parameter Server Training
# --------------------------------------
# We'll now create a synchronous parameter server training scheme. We'll first
# instantiate a process for the parameter server, along with multiple
# workers.

iterations = 50
num_workers = int(sys.argv[1])
staleness_threshold = int(sys.argv[2])

print("Running Stale Ssynchronous Parameter Server Training.")

ray.init(ignore_reinit_error=True)
ps = Server1.ParameterServer.remote(1e-2)
workers = [Server1.Worker.remote() for i in range(num_workers)]

###########################################################################
# Here, workers will  compute the gradients with bounded staleness given its
# current weights and send these gradients to the parameter server as
# soon as they are ready. When the Parameter server finishes applying the
# new gradient, the server will send back a copy of the current weights to the
# worker. The worker will then update the weights and repeat.

###########################################################################
# We'll also instantiate a model on the driver process to evaluate the test
# accuracy during training.

#staleness_threshold = 4

out_file = "SSP{0}_out{1}_acc_loss.txt".format(staleness_threshold, num_workers)
time_file = "SSP{0}_out{1}_iter_time.txt".format(staleness_threshold,num_workers)

f_out = open(out_file, "a")
f_time = open(time_file, "a")

model = Server1.ConvNet()
test_loader = Server1.get_data_loader()[1]

###########################################################################
start_time = time.time()
init_start_time = time.time()
current_weights = ps.get_weights.remote()
permitted_workers = []
# dict for gradient per worker
gradients = {}

# initial gradient computation
for worker in workers:
    gradients[worker.compute_gradients.remote(current_weights)] = worker
print(gradients)

# monitoring iteration
accuracy = 0
i = 0
while accuracy <= 60:
    slowest_time = 10000000000
    for worker in workers:
        c_time = ray.get(worker.get_time_step.remote())
        if c_time <= slowest_time:
            slowest_time = c_time
            slowest_worker = worker

    ready_gradient_list, _ = ray.wait(list(gradients))
    # print(ready_gradient_list)
    if len(ready_gradient_list) > 0:
        ready_gradient_id = ready_gradient_list[0]
        worker = gradients.pop(ready_gradient_id)
        print(worker)
    # Compute and apply gradients

    # add new gradients to dict
    if (ray.get(worker.get_time_step.remote()) - slowest_time) < staleness_threshold:
        current_weights = ps.apply_gradients.remote(*[ready_gradient_id])
        gradients[worker.compute_gradients.remote(current_weights)] = worker
        #print(gradients)
    else:
        print("Worker {0} going too fast and skipped".format(worker))
        try:
            gradients[slowest_worker.compute_gradients.remote(current_weights)] = slowest_worker
            ready_gradient_id = list(gradients.keys())[list(gradients.values()).index(slowest_worker)]
            current_weights = ps.apply_gradients.remote(*[ready_gradient_id])
            print(ready_gradient_id)
        except:
            print("Nothing to do here")

    max_time = 0
    done = False
    for worker in workers:
        #print("Worker {0} is on iteration {1}".format(worker, ray.get(worker.get_time_step.remote())))
        if ray.get(worker.get_time_step.remote()) % 2000 == 0 and done == False:
            max_time = ray.get(worker.get_time_step.remote())
            f_time.write("{} {:.5f}".format(max_time, time.time() - start_time))
            # Evaluate the current model.
            model.set_weights(ray.get(current_weights))
            loss, accuracy = Server1.evaluate(model, test_loader)
            print("Iter {}: \taccuracy is {:.3f} \tloss is {:.6f} \t Iteration {}".format(
                max_time, accuracy, loss, i))
            # end_time = time.time()
            f_out.write('{} {:3f} {:6f} {}\n'.format(max_time, accuracy, loss, i))
            f_time.write(" {:.3f}\n".format(accuracy))
            done = True
            start_time = time.time()
    i += 1

#model.set_weights(ray.get(current_weights))
#loss, accuracy = Server1.evaluate(model, test_loader)
#print("Iter {}: \taccuracy is {:.3f} \tloss is {:.6f}".format(i, accuracy, loss))
end_time = time.time()
f_time.write("{:.5f}".format(time.time() - init_start_time))
print("Final Runtime is: {0}".format(time.time() - init_start_time))
print("Final accuracy is {:.3f}.".format(accuracy))
print("Final loss is {:.6f}.".format(loss))
ray.shutdown()
print("Distributed SSP training completed with {0} workers".format(num_workers))
f_out.close()
f_time.close()