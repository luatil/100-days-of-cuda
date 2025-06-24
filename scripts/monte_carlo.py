import sys

def main():
    lines = sys.stdin.read()
    lines = lines.split("\n")
    rng = (lines[0].split(" "))
    a, b = float(rng[0]), float(rng[1])
    values = [float(el) for el in lines[2].split(" ") if el != ""]
    print(sum(values)*(1/len(values))*(b-a))


if __name__=="__main__":
    main()

