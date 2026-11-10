package graphwarrl;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Line-oriented normal-mode trajectory and collision oracle.
 */
public final class HeadlessSimulator {
    private static final int PLANE_WIDTH = 770;
    private static final int PLANE_HEIGHT = 450;
    private static final double GAME_WIDTH = 50.0;
    private static final int SOLDIER_RADIUS = 7;
    private static final double STEP_SIZE = 0.01;
    private static final int MAX_STEPS = 20_000;
    private static final double MAX_STEP_DISTANCE_SQUARED = 0.001;
    private static final double MIN_X_STEP_DISTANCE = 0.00001;
    private static final double ANGLE_ERROR = Math.PI / 360.0;
    private static final int MAX_ANGLE_LOOPS = 100;

    private record Soldier(int id, int team, double x, double y, boolean alive) {}

    private record Circle(int x, int y, int radius) {}

    private record Point(double x, double y) {}

    private record TraceResult(
        List<Point> trajectory,
        List<Integer> hitIds,
        Double closestEnemyDistance,
        Point collisionPoint
    ) {}

    private static final class Node {
        private final int token;
        private final double value;
        private final Node[] children;

        private Node(int token, double value, Node[] children) {
            this.token = token;
            this.value = value;
            this.children = children;
        }
    }

    private static final class Decoder {
        private final int[] tokens;
        private final double[] constants;
        private int tokenIndex;
        private int constantIndex;

        private Decoder(int[] tokens, double[] constants) {
            this.tokens = tokens;
            this.constants = constants;
        }

        private Node decode() {
            if (tokenIndex >= tokens.length) {
                throw new IllegalArgumentException("Incomplete equation tokens");
            }
            int token = tokens[tokenIndex++];
            int arity = arity(token);
            double value = Double.NaN;
            if (token == 15) {
                if (constantIndex >= constants.length) {
                    throw new IllegalArgumentException("Missing equation constant");
                }
                value = constants[constantIndex++];
                if (!Double.isFinite(value)) {
                    throw new IllegalArgumentException("Non-finite equation constant");
                }
            }
            Node[] children = new Node[arity];
            for (int childIndex = 0; childIndex < arity; childIndex++) {
                children[childIndex] = decode();
            }
            return new Node(token, value, children);
        }

        private Node decodeComplete() {
            Node root = decode();
            if (tokenIndex != tokens.length || constantIndex != constants.length) {
                throw new IllegalArgumentException("Trailing equation data");
            }
            return root;
        }
    }

    private HeadlessSimulator() {}

    private static int arity(int token) {
        return switch (token) {
            case 0, 2, 3, 4 -> 2;
            case 1, 5, 6, 7, 8, 9, 10, 11 -> 1;
            case 12, 13, 14, 15 -> 0;
            default -> throw new IllegalArgumentException("Unknown token: " + token);
        };
    }

    private static double evaluate(Node node, double xValue) {
        return switch (node.token) {
            case 0 -> evaluate(node.children[0], xValue) + evaluate(node.children[1], xValue);
            case 1 -> -evaluate(node.children[0], xValue);
            case 2 -> evaluate(node.children[0], xValue) * evaluate(node.children[1], xValue);
            case 3 -> evaluate(node.children[0], xValue) / evaluate(node.children[1], xValue);
            case 4 -> Math.pow(
                evaluate(node.children[0], xValue),
                evaluate(node.children[1], xValue)
            );
            case 5 -> Math.sqrt(evaluate(node.children[0], xValue));
            case 6 -> Math.log10(evaluate(node.children[0], xValue));
            case 7 -> Math.abs(evaluate(node.children[0], xValue));
            case 8 -> Math.sin(evaluate(node.children[0], xValue));
            case 9 -> Math.cos(evaluate(node.children[0], xValue));
            case 10 -> Math.tan(evaluate(node.children[0], xValue));
            case 11 -> Math.log(evaluate(node.children[0], xValue));
            case 12 -> xValue;
            case 13, 14 -> Double.NaN;
            case 15 -> node.value;
            default -> Double.NaN;
        };
    }

    private static double startAngle(Node equation, double startX, double radius) {
        double initial = evaluate(equation, startX);
        double following = evaluate(equation, startX + STEP_SIZE);
        if (!Double.isFinite(initial) || !Double.isFinite(following)) {
            return Double.NaN;
        }
        double angle = Math.atan((following - initial) / STEP_SIZE);
        double error = Double.POSITIVE_INFINITY;
        int loopCount = 0;
        while (error > ANGLE_ERROR && loopCount < MAX_ANGLE_LOOPS) {
            double finalX = startX + radius * Math.cos(angle);
            double current = evaluate(equation, finalX);
            following = evaluate(equation, finalX + STEP_SIZE);
            if (!Double.isFinite(current) || !Double.isFinite(following)) {
                return Double.NaN;
            }
            double newAngle = Math.atan((following - current) / STEP_SIZE);
            error = Math.abs(newAngle - angle);
            angle = newAngle;
            loopCount++;
        }
        return angle;
    }

    private static Point gameToPixel(double gameX, double gameY, boolean inverted) {
        double pixelX = PLANE_WIDTH * gameX / GAME_WIDTH + PLANE_WIDTH / 2.0;
        double pixelY = -PLANE_WIDTH * gameY / GAME_WIDTH + PLANE_HEIGHT / 2.0;
        if (inverted) {
            pixelX = PLANE_WIDTH - pixelX;
        }
        return new Point(pixelX, pixelY);
    }

    private static boolean blocked(
        int xCoordinate,
        int yCoordinate,
        List<Circle> obstacles,
        List<Circle> explosions
    ) {
        boolean insideObstacle = false;
        for (Circle obstacle : obstacles) {
            long deltaX = xCoordinate - obstacle.x;
            long deltaY = yCoordinate - obstacle.y;
            if (deltaX * deltaX + deltaY * deltaY <= (long) obstacle.radius * obstacle.radius) {
                insideObstacle = true;
                break;
            }
        }
        if (!insideObstacle) {
            return false;
        }
        for (Circle explosion : explosions) {
            long deltaX = xCoordinate - explosion.x;
            long deltaY = yCoordinate - explosion.y;
            if (deltaX * deltaX + deltaY * deltaY <= (long) explosion.radius * explosion.radius) {
                return false;
            }
        }
        return true;
    }

    private static TraceResult trace(
        Node equation,
        int shooterId,
        int shooterTeam,
        double shooterX,
        double shooterY,
        List<Soldier> soldiers,
        List<Circle> obstacles,
        List<Circle> explosions,
        int maximumPoints
    ) {
        boolean inverted = shooterTeam == 2;
        double startPixelX = inverted ? PLANE_WIDTH - shooterX : shooterX;
        double startGameX = GAME_WIDTH * (startPixelX - PLANE_WIDTH / 2.0) / PLANE_WIDTH;
        double startGameY = GAME_WIDTH * (-shooterY + PLANE_HEIGHT / 2.0) / PLANE_WIDTH;
        double gameRadius = GAME_WIDTH * SOLDIER_RADIUS / PLANE_WIDTH;
        double fireAngle = startAngle(equation, startGameX, gameRadius);
        if (!Double.isFinite(fireAngle)) {
            return null;
        }
        double currentGameX = startGameX + gameRadius * Math.cos(fireAngle);
        double currentGameY = startGameY + gameRadius * Math.sin(fireAngle);
        double functionAtStart = evaluate(equation, currentGameX);
        if (!Double.isFinite(functionAtStart)) {
            return null;
        }
        double offset = -functionAtStart + currentGameY;

        List<Point> points = new ArrayList<>();
        points.add(gameToPixel(currentGameX, currentGameY, inverted));
        List<Integer> hitIds = new ArrayList<>();
        Set<Integer> hitSet = new HashSet<>();
        Double closestEnemyDistance = null;
        Point collisionPoint = null;

        for (int stepIndex = 1; stepIndex < MAX_STEPS; stepIndex++) {
            double candidateStep = STEP_SIZE;
            double nextGameX = currentGameX + candidateStep;
            double nextGameY = evaluate(equation, nextGameX) + offset;
            if (!Double.isFinite(nextGameY)) {
                break;
            }
            while (
                (nextGameX - currentGameX) * (nextGameX - currentGameX)
                + (nextGameY - currentGameY) * (nextGameY - currentGameY)
                > MAX_STEP_DISTANCE_SQUARED
            ) {
                if (nextGameX - currentGameX <= MIN_X_STEP_DISTANCE) {
                    nextGameX = currentGameX;
                    break;
                }
                candidateStep /= 2.0;
                nextGameX = currentGameX + candidateStep;
                nextGameY = evaluate(equation, nextGameX) + offset;
                if (!Double.isFinite(nextGameY)) {
                    break;
                }
            }
            if (nextGameX == currentGameX || !Double.isFinite(nextGameY)) {
                break;
            }

            Point pixel = gameToPixel(nextGameX, nextGameY, inverted);
            points.add(pixel);
            for (Soldier soldier : soldiers) {
                if (!soldier.alive || soldier.id == shooterId) {
                    continue;
                }
                double distance = Math.hypot(soldier.x - pixel.x, soldier.y - pixel.y);
                if (soldier.team != shooterTeam) {
                    closestEnemyDistance = closestEnemyDistance == null
                        ? distance
                        : Math.min(closestEnemyDistance, distance);
                }
                if (distance < SOLDIER_RADIUS && hitSet.add(soldier.id)) {
                    hitIds.add(soldier.id);
                }
            }

            int integerX = (int) pixel.x;
            int integerY = (int) pixel.y;
            if (
                integerX < 0
                || integerX >= PLANE_WIDTH
                || integerY < 0
                || integerY >= PLANE_HEIGHT
            ) {
                break;
            }
            if (blocked(integerX, integerY, obstacles, explosions)) {
                collisionPoint = new Point(integerX, integerY);
                break;
            }
            currentGameX = nextGameX;
            currentGameY = nextGameY;
        }

        if (points.size() > maximumPoints) {
            List<Point> sampled = new ArrayList<>(maximumPoints);
            for (int pointIndex = 0; pointIndex < maximumPoints; pointIndex++) {
                double fraction = (double) pointIndex / (maximumPoints - 1);
                int sourceIndex = (int) Math.round(fraction * (points.size() - 1));
                sampled.add(points.get(sourceIndex));
            }
            points = sampled;
        }
        return new TraceResult(points, hitIds, closestEnemyDistance, collisionPoint);
    }

    private static int[] parseIntegers(String value) {
        if (value.isEmpty()) {
            return new int[0];
        }
        String[] fields = value.split(",");
        int[] parsed = new int[fields.length];
        for (int index = 0; index < fields.length; index++) {
            parsed[index] = Integer.parseInt(fields[index]);
        }
        return parsed;
    }

    private static double[] parseDoubles(String value) {
        if (value.isEmpty()) {
            return new double[0];
        }
        String[] fields = value.split(",");
        double[] parsed = new double[fields.length];
        for (int index = 0; index < fields.length; index++) {
            parsed[index] = Double.parseDouble(fields[index]);
        }
        return parsed;
    }

    private static List<Soldier> parseSoldiers(String value) {
        List<Soldier> soldiers = new ArrayList<>();
        if (value.isEmpty()) {
            return soldiers;
        }
        for (String record : value.split(";")) {
            String[] fields = record.split(",");
            if (fields.length != 5) {
                throw new IllegalArgumentException("Malformed soldier record");
            }
            soldiers.add(
                new Soldier(
                    Integer.parseInt(fields[0]),
                    Integer.parseInt(fields[1]),
                    Double.parseDouble(fields[2]),
                    Double.parseDouble(fields[3]),
                    fields[4].equals("1")
                )
            );
        }
        return soldiers;
    }

    private static List<Circle> parseCircles(String value) {
        List<Circle> circles = new ArrayList<>();
        if (value.isEmpty()) {
            return circles;
        }
        for (String record : value.split(";")) {
            String[] fields = record.split(",");
            if (fields.length != 3) {
                throw new IllegalArgumentException("Malformed circle record");
            }
            circles.add(
                new Circle(
                    Integer.parseInt(fields[0]),
                    Integer.parseInt(fields[1]),
                    Integer.parseInt(fields[2])
                )
            );
        }
        return circles;
    }

    private static String formatPoints(List<Point> points) {
        StringBuilder builder = new StringBuilder();
        for (int index = 0; index < points.size(); index++) {
            if (index > 0) {
                builder.append(';');
            }
            Point point = points.get(index);
            builder.append(Double.toString(point.x));
            builder.append(',');
            builder.append(Double.toString(point.y));
        }
        return builder.toString();
    }

    private static String formatIntegers(List<Integer> values) {
        StringBuilder builder = new StringBuilder();
        for (int index = 0; index < values.size(); index++) {
            if (index > 0) {
                builder.append(',');
            }
            builder.append(values.get(index));
        }
        return builder.toString();
    }

    private static String handle(String request) {
        if (request.equals("PING")) {
            return "PONG";
        }
        String[] fields = request.split("\t", -1);
        if (fields.length != 11 || !fields[0].equals("TRACE")) {
            throw new IllegalArgumentException("Unsupported protocol request");
        }
        Node equation = new Decoder(
            parseIntegers(fields[1]),
            parseDoubles(fields[2])
        ).decodeComplete();
        TraceResult result = trace(
            equation,
            Integer.parseInt(fields[3]),
            Integer.parseInt(fields[4]),
            Double.parseDouble(fields[5]),
            Double.parseDouble(fields[6]),
            parseSoldiers(fields[7]),
            parseCircles(fields[8]),
            parseCircles(fields[9]),
            Integer.parseInt(fields[10])
        );
        if (result == null) {
            return "INVALID";
        }
        String closest = result.closestEnemyDistance == null
            ? "null"
            : Double.toString(result.closestEnemyDistance);
        String collision = result.collisionPoint == null
            ? "null"
            : Integer.toString((int) result.collisionPoint.x)
                + ","
                + Integer.toString((int) result.collisionPoint.y);
        return String.join(
            "\t",
            "OK",
            formatPoints(result.trajectory),
            formatIntegers(result.hitIds),
            closest,
            collision
        );
    }

    /**
     * Run the line-oriented subprocess service.
     *
     * @param arguments unused command-line arguments
     * @throws Exception when standard streams fail
     */
    public static void main(String[] arguments) throws Exception {
        Locale.setDefault(Locale.ROOT);
        try (
            BufferedReader reader = new BufferedReader(
                new InputStreamReader(System.in, StandardCharsets.UTF_8)
            );
            BufferedWriter writer = new BufferedWriter(
                new OutputStreamWriter(System.out, StandardCharsets.UTF_8)
            )
        ) {
            String request;
            while ((request = reader.readLine()) != null) {
                if (request.equals("CLOSE")) {
                    writer.write("BYE");
                    writer.newLine();
                    writer.flush();
                    return;
                }
                try {
                    writer.write(handle(request));
                } catch (RuntimeException error) {
                    String message = error.getMessage() == null
                        ? error.getClass().getSimpleName()
                        : error.getMessage().replace('\t', ' ');
                    writer.write("ERROR\tprotocol\t" + message);
                }
                writer.newLine();
                writer.flush();
            }
        }
    }
}
