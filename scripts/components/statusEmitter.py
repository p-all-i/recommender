import datetime, pika, json, threading, time
import functools
print=functools.partial(print, flush=True)


class NodeCommServer:
    """
    A class responsible for handling message queue communication between nodes.
    
    Attributes:
    -----------
    exchange_subscribe : str
        Name of the exchange to subscribe to
    exchange_publish : str
        Name of the exchange to publish messages
    consuming_queue : str
        Name of the queue to consume messages from
    publishing_queue : str
        Name of the queue to publish messages to
    loggerObj : object
        Logger object to record logs
    
    Methods:
    --------
    start():
        Initializes the server, sets up connections, and declares exchanges and queues.
    read():
        Reads messages from the consuming queue.
    send(message):
        Sends a message to the publishing queue.
    """
    
    def __init__(self, exchange_publish_name, publishing_queue, host,  loggerObj=None):
        """Initializes the NodeCommServer object with the given attributes."""
        # self.exchange_subscribe = exchange_subscribe_name
        self.exchange_publish = exchange_publish_name
        # self.consuming_queue = consuming_queue
        self.publishing_queue = publishing_queue
        # self.routing_keys = routing_keys
        self.loggerObj = loggerObj
        self.host = host
    
    def start(self):
        """
        Initializes the server, sets up connections, and declares exchanges and queues.
        """
        creds = pika.PlainCredentials('guest', 'guest')
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, credentials=creds, heartbeat=15, retry_delay=1, connection_attempts=10))
        self.channel = self.connection.channel()

        # self.channel.exchange_declare(exchange=self.exchange_subscribe, exchange_type='direct', durable=False)
        # self.channel.queue_declare(queue=self.consuming_queue, durable=False, arguments={'x-message-ttl': 30000})
        # # Binding the queues with multiple routing keys
        # for rounting_key in self.routing_keys:
        #     self.channel.queue_bind(exchange=self.exchange_subscribe, queue=self.consuming_queue, routing_key=rounting_key)
        
        self.channel.exchange_declare(exchange=self.exchange_publish, exchange_type='direct')
        self.channel.queue_declare(queue=self.publishing_queue, durable=False, arguments={'x-message-ttl': 30000})
        self.channel.queue_bind(exchange=self.exchange_publish, queue=self.publishing_queue, routing_key=self.publishing_queue)
        print(f"[INFO] {datetime.datetime.now()} Connected with RabbitMQ Qs!!!")

    def send(self, message):
        """
        Sends a message to the publishing queue.
        
        Parameters:
        -----------
        message : dict
            The message to be sent.
        """
        message_json = json.dumps(message)
        # self.channel.basic_publish(exchange=self.exchange_publish, routing_key=self.publishing_queue, body=message_json)
        while True:
            try:
                self.channel.basic_publish(exchange=self.exchange_publish, routing_key=self.publishing_queue, body=message, properties=pika.BasicProperties(
                          content_type='application/json',
                      ))
                break
            except Exception as e:
                self.start()
        print(f"[INFO] {datetime.datetime.now()} Publishing result for to the result Queue")
        # self.loggerObj.queuing_logger.info(f"Publishing result for {message['cameraId']} to the result Queue")
        # self.loggerObj.loop_logger.info(f"Publishing result for {message['cameraId']} to the result Queue")





# Class responsible for sending training updates
class ControlledThread(threading.Thread):
    def __init__(self, messenger, config_id, STATUS):
        super(ControlledThread, self).__init__()
        self.running = True
        self.messenger = messenger
        self.config_id = config_id
        self.TRAINING_STATUS = STATUS

    def run(self):
        # global currentEpoch  # Indicate that we're using the global variable
        while self.running:
            message = json.dumps({"config_id": self.config_id, "status": self.TRAINING_STATUS})
            self.messenger.send(message)
            print(f"Recommendention IS RUNNING")
            if not self.running:
                break
            
            time.sleep(5)  # sending every 5 seconds

    def stop(self):
        self.running = False