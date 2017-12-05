/**
 * Created by hb-laptop on 2017-04-21.
 */
import java.net.*;
import java.io.*;
import java.nio.charset.StandardCharsets;

public class EchoServer
{
    public static void main(String[] args) throws IOException
    {
        ServerSocket serverSocket = null;

        try {
            serverSocket = new ServerSocket(10007);
        }
        catch (IOException e)
        {
            System.exit(1);
        }

        Socket clientSocket = null;        
        while (true) {
            try {
                clientSocket = serverSocket.accept();
                new sdes(clientSocket).start();
            }
            catch (IOException e)
            {
                System.err.println("Accept failed.");
                System.exit(1);
            }
        }
    }
    private static class sdes extends Thread {

        private Socket socket;
        private OutputStream out;
        private DataOutputStream dos;
        private BufferedReader bin;
        private String host;
        private PrintWriter pw;

        public sdes(Socket sck) {
            this.socket = sck;
            this.host = sck.getInetAddress().toString();
        }
        public void run() {
            try {
                this.out = socket.getOutputStream();
                this.dos = new DataOutputStream(out);
                this.pw = new PrintWriter(out, true);
                this.bin = new BufferedReader(new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8));

                String inputLine;
                while ((inputLine = bin.readLine()) != null)
                {
                    System.out.println(inputLine);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }finally {
                try {
                    out.close();
                    dos.close();
                    pw.close();
                    bin.close();
                }catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }
}

