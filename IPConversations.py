""" Classes to support statistical analysis of IP conversations

These were developed specifically to automate the identification of candidate
conversations suggestive of a compromised host beaconing to a C&C server.
Identification is based on a score assuming that:
    1.) Beaconing conversations will continue throughout the entire capture
        window
    2.) Beaconing conversations will have a clearly defined periodicity, which
        will manifest as a non-flat power-spectrum produced by Fourier analysis
        of the packet time histogram.

All identification is perfomred at the IP layer. TCP/UDP port numbers are not
used to seperate conversations between the same pair of IP addresses.

Scapy is used for the packet operations. The performance of these scripts is
very poor, particulary in loading PCAP files and filtering packet lists. It is
unclear is Scapy is the cause of the poor performance, or if I am just using
the API inefficiently.

Example usage is:
a = IPConversations.IPConversationListFixedEndpoint("172.16.106.129",
                                                    filename="challenge.pcap")
a.print_sorted_conversation_stats()
a.plot_stats()
a.get_ip_conversation('172.16.106.200').plot_histogram()
a.get_ip_conversation('172.16.106.200').print_summary()
a.get_ip_conversation('172.16.106.200').print_hexdump()
"""

import scapy.all
import scapy.utils
import numpy
import numpy.fft
import matplotlib.pyplot as pyplot

def _match_ip(packet, ip_addr):
    """ Find if packet matches specified IP address

    Parameters
    ----------
    packet : Scapy packet object
    ip_addr : String
        IP address using decimal octets (i.e. '255.255.255.0')

    Returns
    -------
    Boolean
    """
    if not(packet.haslayer('IP')):
        return False
    if (packet.getlayer('IP').src == ip_addr or 
            packet.getlayer('IP').dst == ip_addr):
        return True
    else:
        return False

class IPConversation():
    """ Class describing a conversation between two IP endpoints"""
    def __init__(self, packets, ip_addr1, ip_addr2):
        """ Initialize
        
        Parameters
        ----------
        packets : scapy.plist.PacketList object
        ip_addr1 : String
            Address of one IP endpoint in decimal octets
        ip_addr2 : String
            Address of other IP endpoint in decimal octets

        Returns
        -------
        None
        """
        self.packets = packets
        self.ip_addrs = [ip_addr1, ip_addr2]
        self.packet_timeseries = None

    def get_ip_addrs(self):
        """ Get IP addresses of endpoints in this conversation

        Parameters
        ----------
        None

        Returns
        -------
        List of two strings with IP addresses in decimal octets
        """
        return self.ip_addrs

    def get_other_partner_addr(self, ip_addr):
        """ Get other IP endpoint given one endpoint

        Parameters
        ----------
        ip_addr : String
            Address of one IP endpoint in decimal octets

        Returns
        -------
        String containing IP address of other endpoint in decimal octets
        """
        if self.ip_addrs[0] != ip_addr and self.ip_addrs[1] != ip_addr:
            raise ValueError(f"{ip_addr} not a partner in conversation")
        if self.ip_addrs[0] == ip_addr:
            return self.ip_addrs[1]
        else:
            return self.ip_addrs[0]

    def get_packet_timeseries(self):
        """ Get times associated with all packets in conversation

        All times are relative to the start of the conversation

        Parameters
        ----------
        None

        Returns
        -------
        List of floats corresponding to time offset of each packet from start
        of conversation
        """
        if self.packet_timeseries:
            return self.packet_timeseries

        self.packet_timeseries = []
        start = self.packets[0].time
        for p in self.packets:
            self.packet_timeseries.append(float(p.time - start))
        return self.packet_timeseries

    def get_duration(self):
        """ Get the total duration of the conversation

        Parameters
        ----------
        None

        Returns
        -------
        float representing total conversation time
        """
        ts = self.get_packet_timeseries()
        return ts[-1] - ts[0]

    def calculate_timeseries_histogram(self, interval=1.0):
        """ Calculate the histogram of packet times

        Parameters
        ----------
        interval : float, optional
            Histogram window width in seconds

        Returns
        -------
        tuple of numpy.ndarrays where the first ndarray is the number of
        counts in each window and the second ndarray gives the edges of the
        histogram windows in seconds
        """
        # Override poor interval choices to ensure that we have at least 10
        # windows There may still be problems if a conversation consists of a
        # single packet.
        if self.get_duration()/interval < 10:
            interval = self.get_duration()/10.0
        ts = self.get_packet_timeseries()
        return numpy.histogram(ts,
                               numpy.arange(0, round(ts[-1] + 2), interval))

    def get_timeseries_fft(self, interval=1.0):
        """ Calculate an FFT of the histogrammed timeseries

        Parameters
        ----------
        interval : float, optional
            Histogram window width in seconds

        Returns
        -------
        tuple of numpy.ndarrays where the first tuple is the fft frequencies,
        and the second tuple is the FFT value at that frequency
        """
        # Override poor interval choices to ensure that we have at least 10
        # windows There may still be problems if a conversation consists of a
        # single packet.
        if self.get_duration()/interval < 10:
            interval = self.get_duration()/10.0
        samples = self.calculate_timeseries_histogram(interval=interval)[0]
        if len(samples) == 0:
            raise ValueError(f"Timeseries histogram of length zero: {self.get_ip_addrs()}")
        return (numpy.fft.fftfreq(len(samples), d=interval),
                numpy.fft.fft(samples))

    def get_timeseries_powerspectrum(self, interval=1.0):
        """ Calculate a power spectrum of the histogrammed timeseries

        Parameters
        ----------
        interval : float, optional
            Histogram window width in seconds

        Returns
        -------
        tuple of numpy.ndarrays where the first tuple is the FFT frequencies,
        and the second tuple is the power spectrum value at that frequency
        """
        # Override poor interval choices to ensure that we have at least 10
        # windows There may still be problems if a conversation consists of a
        # single packet.
        if self.get_duration()/interval < 10:
            interval = self.get_duration()/10.0
        (freq, f) = self.get_timeseries_fft(interval=interval)
        return (freq, f*f.conj())

    def get_spectral_flatness(self, interval=1.0):
        """ Calculate spectral flatness of the histogrammed timeseries

        Calculates the spectral flatness, also known as the Wiener entropy,
        from the power spectrum. This is the ratio of the geometric mean
        of the power series to the arithmetic mean. For a flat spectrum,
        this value will be 1. The value approaches zero as the spectrum
        deviates more from a flat spectrum.

        Parameters
        ----------
        interval : float, optional
            Histogram window width in seconds

        Returns
        -------
        float
        """
        # Override poor interval choices to ensure that we have at least 10
        # windows There may still be problems if a conversation consists of a
        # single packet.
        if self.get_duration()/interval < 10:
            interval = self.get_duration()/10.0
        (freq, ps) = self.get_timeseries_powerspectrum(interval=interval)
        x = numpy.log(ps)
        return numpy.abs(numpy.exp(x.mean())/ps.mean())

    def get_num_packets(self):
        """ Return the number of packets in the conversation

        Parameters
        ----------
        None

        Returns
        -------
        int
        """
        return len(self.packets)

    def plot_histogram(self, interval=1.0, ax=None):
        """ Plot histogrammed packet timestamps

        Parameters
        ----------
        interval : float, optional
            Histogram window width in seconds
        ax : pyplot.ax object, optional
            Axes to which to add the plot

        Returns
        -------
        pyplot.ax object
        """
        if ax is None:
            fig = pyplot.figure()
            ax = fig.add_subplot(1,1,1)
        (samples, bins) = self.calculate_timeseries_histogram(interval=interval)
        ax.plot(bins[:-1], samples)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Packets")
        return ax

    def plot_powerspectrum(self, interval=1.0, ax=None):
        """ Plot power spectrum of histogrammed timestamps

        Parameters
        ----------
        interval : float, optional
            Histogram window width in seconds
        ax : pyplot.ax object, optional
            Axes to which to add the plot

        Returns
        -------
        pyplot.ax object
        """
        if ax is None:
            fig = pyplot.figure()
            ax = fig.add_subplot(1,1,1)
        ax.plot(*self.get_timeseries_powerspectrum(interval=interval))
        ax.set_xlabel("Frequency [rad/s]")
        ax.set_ylabel("Power [arb.]")
        ax.set_yscale('log')
        return ax
    
    def print_summary(self):
        """ Print summary of packets """
        self.packets.summary()

    def print_hexdump(self):
        """ Print hexdump of packets """
        self.packets.hexdump()

class IPConversationListFixedEndpoint():
    """ Class containing information about a number of conversations with
    a single defined IP endpoint
    """
    def __init__(self, ip_addr, packets = None, filename=None):
        """ Initialize

        Parameters
        ----------
        ip_addr : string
            IP endpoint participating in all conversations. String contains four
            decimal octets, i.e. '255.255.255.0'
        packets : scapy.plist.PacketList object, optional
            packet list containing packets of all conversations
        filename : string, optional
            File name of a PCAP file from which the packets should be read

        Returns
        -------
        None
        """
        self.ip_addr = ip_addr
        self.ip_partners = None
        self.ip_conversations = None
        self.raw_packets = None
        self.filtered_packets = None
        if filename:
            packets = scapy.utils.rdpcap(filename)
        if packets:
            self.assign_packets(packets)
        
    def assign_packets(self, packets):
        """ Assign packet list to the object

        Parameters
        ----------
        packets : scapy.plist.PacketList object
            packet list containing packets of all conversations

        Returns
        -------
        None
        """
        self.raw_packets = packets
        # Invalidate any previously generated information
        self.ip_partners =  None
        self.ip_conversations = {}
        self.filtered_packets = None
        
    def get_filtered_packets(self):
        """ Produce filtered list of packets matching specified IP address

        Parameters
        ----------
        None
        
        Returns
        -------
        Filtered scapy.plist.PacketList object
        """
        # Look for cached version
        if self.filtered_packets:
            return self.filtered_packets
        # Otherwise, generate
        f = lambda p: _match_ip(p, self.ip_addr)
        self.filtered_packets = self.raw_packets.filter(f)
        return self.filtered_packets

    def get_ip_partners(self):
        """ Produced filtered list of conversation partner IP addresses

        Parameters
        ----------
        None

        Returns
        -------
        List of strings containing conversation partner IP addresses
        """
        # Return cached version if it exists
        if self.ip_partners:
            return self.ip_partners

        fp = self.get_filtered_packets()
        ips = []
        for p in fp:
            dst = p.getlayer('IP').dst
            src = p.getlayer('IP').src
            # Convert to binary form initially so we can do sorting later
            if dst != self.ip_addr:
                ips.append(socket.inet_pton(socket.AF_INET, dst))
            if src != self.ip_addr:
                ips.append(socket.inet_pton(socket.AF_INET, src))
            # Only append ip_addr if there are packets with ip_addr as both src
            # and dst. Presumably such traffic would not typically appear
            # in a capture.
            if dst == src and dst == self.ip_addr:
                ips.append(socket.inet_pton(socket.AF_INET, dst))
        # Remove duplicates
        ips = list(set(ips))
        ips.sort()

        self.ip_partners = [socket.inet_ntop(socket.AF_INET, ip) for ip in ips]
        return self.ip_partners

    def get_ip_conversation(self, ip_addr):
        """ Get an IPConversation with a defined endpoint

        Note that the second endpoint is the fixed endpoint that applies to
        all conversations in this object.

        Parameters
        ----------
        ip_addr : string
            Contains IP address of target endpoint in decimal octets

        Returns
        -------
        IPConversation object for two defined endpoints
        """
        # Return this directly if we've previously calculated and cached it
        if ip_addr in self.ip_conversations:
            return self.ip_conversations[ip_addr]
        # Filter both on the object's IP address and the requested partner IP
        # address
        fp = self.get_filtered_packets()
        f = lambda p: _match_ip(p, ip_addr)
        # Cache the IPConversation object in a dictionary
        self.ip_conversations[ip_addr] = IPConversation(fp.filter(f),
                                                        self.ip_addr,
                                                        ip_addr)
        return self.ip_conversations[ip_addr]
 
    def get_all_ip_conversations(self):
        """ Return list of all IPConversations in this class

        Parameters
        ----------
        None

        Returns
        -------
        List of IPConversation objects
        """
        return [self.get_ip_conversation(ip) for ip in self.get_ip_partners()]

    def get_duration(self):
        """ Get the total duration of the conversation

        Parameters
        ----------
        None

        Returns
        -------
        float representing total conversation time
        """
        fp = self.get_filtered_packets()
        return float(fp[-1].time - fp[0].time)

    def get_conversation_stats(self, interval=1.0):
        """ Get statistics for conversations

        Statistics include:
        fractional duration : The fraction of the total capture time that a
           conversation between two endpoints endures
        flatness : The spectral flatness of the histogrammed timeseries
           corresponding to a particular conversation
        beacon score : Calculated as 0.5*(fraction duration) + 0.5*(1-
           flatness). An ideal beacon would have a score of 1.

        Parameters
        ----------
        interval : float, optional
            Histogram window width in seconds

        Returns
        -------
        Structured numpy.ndarray with an entry for each conversation. Elements
        of each structured entry are:
           ip : string containing IP address of second endpoint in decimal octets
           frac_duration : float containing fractional duration of conversation
           flatness : float containing spectral flatness
           num_packets : number of packets in the conversation
           beacon_score : calculated beacon score
        """
        stats = numpy.zeros(len(self.get_ip_partners()),
                            dtype = [('ip', 'S15'),
                                     ('frac_duration', 'f8'),
                                     ('flatness', 'f8'),
                                     ('num_packets', 'i8'),
                                     ('beacon_score', 'f8')])
        
        i = 0
        for conv in self.get_all_ip_conversations():
            stats[i]['ip'] = conv.get_other_partner_addr(self.ip_addr)
            stats[i]['frac_duration'] = conv.get_duration()/self.get_duration()
            stats[i]['flatness'] = conv.get_spectral_flatness(interval=interval)
            stats[i]['num_packets'] = conv.get_num_packets()
            stats[i]['beacon_score'] = ((1.0 - stats[i]['flatness'])*0.5 +
                                        stats[i]['frac_duration']*0.5)
            i += 1
        return stats

    def print_sorted_conversation_stats(self, interval=1.0):
        """ Print sorted conversation stats

        Statistics are generated by get_conversation_stats() above

        Parameters
        ----------
        interval : float, optional
            Histogram window width in seconds

        Returns
        -------
        None
        """
        a = numpy.sort(self.get_conversation_stats(interval=interval),
                       order='beacon_score')
        print(a)

    def plot_stats(self, interval=1.0, ax=None):
        """ Plot each conversation on axes of fractional duration and flatness

        Parameters
        ----------
        interval : float, optional
            Histogram window width in seconds
        ax : pyplot.ax object, optional
            Axes to which to add the plot

        Returns
        -------
        pyplot.ax object
        """
        if ax is None:
            fig = pyplot.figure()
            ax = fig.add_subplot(1,1,1)
        stats = self.get_conversation_stats(interval=interval)
        x = []
        y = []
        label = []
        for s in self.get_conversation_stats(interval=interval):
            ax.annotate(s['ip'].decode(), (s['frac_duration'], s['flatness']))
        # Extra wide horizontal access for IP addresses that extend
        ax.set_xlim(0, 1.3)
        ax.set_ylim(0,1)
        ax.set_xlabel("Fractional conversation duration")
        ax.set_ylabel("Spectral flatness")
        return ax

