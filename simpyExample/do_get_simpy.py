import simpy

env = simpy.Environment()


class ConditionalGet(simpy.resources.base.Get):
    def __init__(self, resource, condition=lambda: True):
        self.condition = condition
        super().__init__(resource)


class aStore(simpy.resources.store.Store):
    get = simpy.core.BoundClass(ConditionalGet)
    def _do_get(self, event):
        if event.condition():
            super()._do_get(event)


q1 = aStore(env, capacity=4)
q2 = aStore(env, capacity=4)

def putter():
    i = 0
    while i<5:
        yield env.timeout(1)
        yield q1.put(i)
        print(env.now, 'putter put %d into q1' % i)
        i += 1


def mover():
    while True:
        yield env.timeout(20)
        print(env.now, 'mover waiting to get from q1')
        item = yield q1.get(lambda: len(q2.items) < q2.capacity)
        print(env.now, 'mover got from q1')
        print(env.now, 'mover waiting to put into q2')
        yield q2.put(item)
        print(env.now, 'mover put %d into q2' % item)


def getter():
    while True:
        yield env.timeout(10)
        #print(env.now, 'getter waiting to get from q2')
        a = yield q1.get()
        print(a)
        #print(env.now, 'getter got from q2')


env.process(putter())
#env.process(mover())
env.process(getter())

env.run()