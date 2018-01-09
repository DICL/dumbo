from srm.settings import *


class ContainerGroup(object):
    def __init__(self, containers=None, latencyCritical=False):
        """Initialize Container Group"""
        assert containers
        self.debug = DEBUG
        self.containers = containers
        self.id = self.containers[0].id
        self.latencyCritical = latencyCritical
        
        status = containers[0].status
        if status == 'running':
            self.enabled = True
        else:
            self.enabled = False

    def log(self, fmt):
        """Print debugging log"""
        if self.debug:
            print("[{}] {}".format(self, fmt))

    def update(self, type, value):
        """Update the resource of this container"""
        if type == RESOURCE_CORE:
            for container in self.containers:
                self.log("Update cpuset: {}".format(value))
                container.update(cpuset_cpus=str(value))
        else:
            raise RuntimeError("Invalid resource type")

    def enable(self):
        """Unpause this container"""
        for container in self.containers:
            if container.status == 'exited':
                container.start()
                container.attrs['State']['Status'] = 'running'
            elif container.status == 'paused':
                container.unpause()
                container.attrs['State']['Status'] = 'running'
            self.log("enable")
            self.enabled = True

    def disable(self):
        """Pause this container"""
        for container in self.containers:
            if container.status == 'running':
                container.pause()
                container.attrs['State']['Status'] = 'paused'
            self.log("disable")
            self.enabled = False

    def __str__(self):
        """Override the default str behavior"""
        return "{}CG".format(
            "LC" if self.latencyCritical else "BA",
        )

    def __eq__(self, other):
        """Override the default equals behavior"""
        if isinstance(other, self.__class__):
            return self.id == other.id
        return False

    def __ne__(self, other):
        """Define a non-equality test"""
        return not self.__eq__(other)

    def __hash__(self):
        """Override the default hash behavior"""
        return hash(self.id)
