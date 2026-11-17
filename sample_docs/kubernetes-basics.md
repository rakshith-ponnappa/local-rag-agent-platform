# Kubernetes Basics

A Pod is the smallest deployable unit in Kubernetes. It wraps one or more containers that share a network namespace and storage volumes.

A Deployment manages a ReplicaSet, which keeps the desired number of Pod replicas running. Rolling updates replace Pods gradually so the application stays available during a release.

A Service gives a stable virtual IP and DNS name to a changing set of Pods selected by labels. ClusterIP is internal only, NodePort opens a port on every node, and LoadBalancer provisions an external load balancer.

Liveness probes restart a container that is stuck. Readiness probes remove a Pod from Service endpoints until it is ready to receive traffic.

Use kubectl get pods to list Pods and kubectl describe pod to see events such as image pull errors or failed probes.
