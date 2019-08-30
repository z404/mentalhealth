import java.util.*;
public class javacode {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();
        while (t != 0 && (t >= 1 && t <= 1000)) {
            sc.useDelimiter("\n");
            String mn = sc.next();
            String s[] = mn.split(" ");
            int m = Integer.parseInt(s[0]);
            int n = Integer.parseInt(s[1]);
            if ((n >= 1 && n <= 100) && (m >= 1 && m <= 100)) {
                int A[][] = new int[m][n];
                int B[][] = new int[m][n];
                for (int i = 0; i < m; i++) {
                    String a = sc.next();
                    for (int j = 0; j < n; j++) {
                        A[i][j] = Character.getNumericValue(a.charAt(j));
                    }
                }
                for (int i = 0; i < m; i++) {
                    String a = sc.next();
                    for (int j = 0; j < n; j++) {
                        B[i][j] = Character.getNumericValue(a.charAt(j));
                    }
                }
                int c = 0, c1 = 0;
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        c = c + A[i][j];
                        c1 = c1 + B[i][j];
                    }
                }
                int bool = 0;
                for (int i = 0; i < m; i++) {
                    if (Arrays.equals(A[i], B[i])) {
                        bool += 1;
                    }
                }
                if ((c == c1) && bool != m) {
                    int num = 0;
                    for (int i = 0; i < m; i++) {
                        for (int j = 0; j < n; j++) {
                            if (A[i][j] != B[i][j]) {
                                num += 1;
                            }
                        }
                    }
                    int val[] = new int[num];
                    int row[] = new int[num];
                    int col[] = new int[num];
                    int l = 0;
                    for (int i = 0; i < m; i++) {
                        for (int j = 0; j < n; j++) {
                            if (A[i][j] != B[i][j]) {
                                val[l] = A[i][j];
                                row[l] = i;
                                col[l] = j;
                                l = l + 1;
                            }
                        }
                    }
                    Arrays.sort(val);
                    int count = 0, d = 0, len = val.length - 1;
                    for (int i = 0; i < val.length / 2; i++) {
                        int temp = val[i];
                        val[i] = val[len];
                        val[len] = temp;
                        temp = row[i];
                        row[i] = row[len];
                        row[len] = temp;
                        temp = col[i];
                        col[i] = col[len];
                        col[len] = temp;
                        len -= 1;
                        d = i;
                    }
                    count = d + 1;
                    for (int i = 0; i < val.length; i++) {
                        A[row[i]][col[i]] = val[i];
                    }
                    System.out.println(count);
                } else if (bool == m) {
                    System.out.println("0");
                } else {
                    System.out.println("-1");
                }
            }
            t -= 1;
        }
    }
}
