MAXSIZE = 100000 #enough for >300x300 image size (which is, like the max currently of the aerofoil after shrinkage)

class Queue(object):

    def __init__(self):
        self.queue = []
        self.head = -1
        self.tail = -1
        self.max_size = MAXSIZE
    
    def enqueue(self, item):
        queue_length = self.get_length()
        if queue_length == self.max_size:
            print("queue full", queue_length)
        else:
            if queue_length == 0:
                self.update_head(1)
                self.update_tail(1)
            else:
                if self.tail == self.max_size-1:
                    self.update_tail(-self.tail)
                else:
                    self.update_tail(1)    
            self.add_to_queue(item)
        
    def dequeue(self):
        queue_length = self.get_length()
        if queue_length == 0:
            print("queue empty")
            return -1
        else:
            item = self.queue[self.head]
            if queue_length == 1:
                self.update_head(-(self.head+1))
                self.update_tail(-(self.tail+1))
                self.queue = []
            else:
                if self.head == self.max_size-1:
                    self.update_head(-self.head)
                else:
                    self.update_head(1)
        return item
    
    
    def get_length(self):
        if self.tail != -1 and self.head != -1:
            if self.tail >= self.head:
                return self.tail - self.head +1
            else:
                return self.max_size - (self.head - self.tail) +1
        else:
            return 0
    
    def update_head(self, change):
        self.head += change
        return self.head
        
    def update_tail(self, change):
        self.tail += change
        return self.tail
    
    def add_to_queue(self, item):
        if self.tail >= self.head:
            self.queue.append(item)
        else:
            self.queue[self.tail] = item
    
    def is_empty(self):
        if self.get_length() == 0:
            return True
        else:
            return False
    
