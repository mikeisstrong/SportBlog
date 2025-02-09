import requests
import json
import os
import datetime
import re
import openai
import markdown2
import glob
from zoneinfo import ZoneInfo  # For timezone conversion
from bs4 import BeautifulSoup  # For scraping team standings
from jinja2 import Environment, FileSystemLoader

# -------------------------------
# Configuration and Global Vars
# -------------------------------

# The Odds API configuration
API_KEY = 'e4e9af144c5043923fcb988093791327'  # Replace with your actual API key
BASE_URL = 'https://api.the-odds-api.com/v4'
REGIONS = ['au', 'uk', 'us']
MARKETS = ['h2h']
TARGET_BOOKMAKERS = ['888sport', 'BetMGM', 'Betway', 'DraftKings', 'FanDuel', 'LeoVegas', 'PointsBet']

# Check for the OpenAI API key.
if not os.environ.get("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY environment variable not set.")

# -------------------------------
# Helper Function for Date Filtering
# -------------------------------

def is_event_today(iso_time_str):
    """
    Given an ISO 8601 datetime string (with 'Z' for UTC),
    return True if the event's date in US Eastern timezone is today.
    """
    try:
        # Replace 'Z' with '+00:00' to make the string ISO compliant.
        dt_utc = datetime.datetime.fromisoformat(iso_time_str.replace("Z", "+00:00"))
    except Exception as e:
        print(f"Error parsing datetime string '{iso_time_str}': {e}")
        return False

    # Convert event time to US Eastern time.
    dt_eastern = dt_utc.astimezone(ZoneInfo("America/New_York"))
    # Get current time in Eastern timezone.
    now_eastern = datetime.datetime.now(ZoneInfo("America/New_York"))
    # Compare the date parts.
    return dt_eastern.date() == now_eastern.date()

# -------------------------------
# Odds Analysis Functions
# -------------------------------

def implied_probability(odds):
    """Calculates the implied probability from decimal odds."""
    try:
        return 1 / float(odds)
    except (TypeError, ValueError):
        return 0

def process_event(event):
    """Processes an event to extract and compute odds details."""
    all_odds = {}
    for bookmaker in event.get('bookmakers', []):
        if bookmaker['title'] in TARGET_BOOKMAKERS:
            for market in bookmaker.get('markets', []):
                if market['key'] == 'h2h':
                    for outcome in market.get('outcomes', []):
                        key = f"{outcome['name']}-{bookmaker['title']}"
                        all_odds[key] = outcome['price']
    if not all_odds:
        return None

    home_team_odds = {k: v for k, v in all_odds.items() if event['home_team'] in k}
    away_team_odds = {k: v for k, v in all_odds.items() if event['away_team'] in k}

    if home_team_odds and away_team_odds:
        min_home_odds = min(home_team_odds.values())
        min_home_bookmaker = min(home_team_odds, key=home_team_odds.get).split('-')[1]
        max_home_odds = max(home_team_odds.values())
        max_home_bookmaker = max(home_team_odds, key=home_team_odds.get).split('-')[1]

        min_away_odds = min(away_team_odds.values())
        min_away_bookmaker = min(away_team_odds, key=away_team_odds.get).split('-')[1]
        max_away_odds = max(away_team_odds.values())
        max_away_bookmaker = max(away_team_odds, key=away_team_odds.get).split('-')[1]

        total_implied_probability = implied_probability(min_home_odds) + implied_probability(min_away_odds)

        avg_home_odds = sum(home_team_odds.values()) / len(home_team_odds)
        avg_away_odds = sum(away_team_odds.values()) / len(away_team_odds)
        avg_home_implied = implied_probability(avg_home_odds)
        avg_away_implied = implied_probability(avg_away_odds)

        return {
            "home_team": event['home_team'],
            "away_team": event['away_team'],
            "game_date": event.get('commence_time'),
            "best_home_odds": {
                "odds": min_home_odds,
                "implied_percentage": implied_probability(min_home_odds),
                "bookmaker": min_home_bookmaker
            },
            "worst_home_odds": {
                "odds": max_home_odds,
                "implied_percentage": implied_probability(max_home_odds),
                "bookmaker": max_home_bookmaker
            },
            "best_away_odds": {
                "odds": min_away_odds,
                "implied_percentage": implied_probability(min_away_odds),
                "bookmaker": min_away_bookmaker
            },
            "worst_away_odds": {
                "odds": max_away_odds,
                "implied_percentage": implied_probability(max_away_odds),
                "bookmaker": max_away_bookmaker
            },
            "average_home_odds": {
                "odds": avg_home_odds,
                "implied_percentage": avg_home_implied
            },
            "average_away_odds": {
                "odds": avg_away_odds,
                "implied_percentage": avg_away_implied
            },
            "arbitrage_opportunity": total_implied_probability < 1,
            "total_implied_probability": total_implied_probability
        }
    return None

def get_odds(sport_key):
    """Fetches odds for a given sport key from The Odds API."""
    url = f"{BASE_URL}/sports/{sport_key}/odds"
    params = {
        'apiKey': API_KEY,
        'regions': ','.join(REGIONS),
        'markets': ','.join(MARKETS),
        'oddsFormat': 'decimal',
        'dateFormat': 'iso'
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json() if response.status_code == 200 else []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching odds for {sport_key}: {e}")
        return []

# -------------------------------
# Blog Generation Functions
# -------------------------------

def clean_team_name(name):
    """Cleans team name by removing trailing numbers in parentheses."""
    cleaned = re.sub(r'\(\d+\)$', '', name).strip()
    return cleaned

def fetch_team_standings(debug=False):
    """
    Fetches team standings from Basketball Reference for the NBA 2025 season.
    Extracts stats like wins, losses, win_pct, SRS, points per game (PS/G), and opponent points per game (PA/G).
    Returns a dictionary mapping cleaned team names to their stats.
    """
    url = "https://www.basketball-reference.com/leagues/NBA_2025_standings.html"
    if debug:
        print(f"Fetching team standings from {url}")
    try:
        response = requests.get(url)
    except Exception as e:
        if debug:
            print(f"Error fetching standings: {e}")
        return {}

    if response.status_code != 200:
        if debug:
            print(f"Error fetching standings: HTTP {response.status_code}")
        return {}

    soup = BeautifulSoup(response.text, "html.parser")
    standings = {}

    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            team_cell = row.find("th", {"data-stat": "team_name"})
            if team_cell:
                team_name = team_cell.get_text(strip=True)
                cleaned_name = clean_team_name(team_name)
                if cleaned_name in ["Eastern Conference", "Western Conference"]:
                    continue
                if cleaned_name in standings:
                    if debug:
                        print(f"Skipping duplicate team row: {cleaned_name}")
                    continue

                wins_cell = row.find("td", {"data-stat": "wins"})
                losses_cell = row.find("td", {"data-stat": "losses"})
                win_pct_cell = row.find("td", {"data-stat": "win_loss_pct"})
                srs_cell = row.find("td", {"data-stat": "srs"})
                pts_cell = row.find("td", {"data-stat": "pts_per_g"})
                opp_pts_cell = row.find("td", {"data-stat": "opp_pts_per_g"})

                standings[cleaned_name] = {
                    "wins": wins_cell.get_text(strip=True) if wins_cell else "N/A",
                    "losses": losses_cell.get_text(strip=True) if losses_cell else "N/A",
                    "win_pct": win_pct_cell.get_text(strip=True) if win_pct_cell else "N/A",
                    "srs": srs_cell.get_text(strip=True) if srs_cell else "N/A",
                    "pts_per_g": pts_cell.get_text(strip=True) if pts_cell else "N/A",
                    "opp_pts_per_g": opp_pts_cell.get_text(strip=True) if opp_pts_cell else "N/A"
                }
                if debug:
                    print(f"Standings for {cleaned_name}: {standings[cleaned_name]}")
    if debug:
        print(f"\nFetched standings for {len(standings)} teams.")
    return standings

def rank_games(game_data):
    """Sorts the list of game dictionaries by the delta (closeness) in ascending order."""
    return sorted(game_data, key=lambda x: x["delta"])

def generate_blog_post(ranked_games):
    """
    Uses the OpenAI ChatCompletion API to generate a blog post based on the ranked games.
    The prompt includes detailed instructions on formatting, statistics, and writer attributions.
    """
    prompt = """Overview:
Create a sports blog post titled "Ranking tonight's Games." The blog is broken into three sections based on the probability differences between teams. Each game’s summary is written according to a strict format and style, with contributions from specific writers. An introductory and closing paragraph, both written by the Editor ("Mistro"), frame the game summaries.

General Requirements:
Complete Coverage of Games:
IMPORTANT: Ensure every single game provided in the rankings is included in the final output.
Double-check that no game is omitted. All games must appear in one of the three sections below.
Game Summary Structure:
Each game summary must consist of an intro.
Include one fact about the home team citing a specific statistic.
Include one fact about the away team citing a specific statistic.
Include 2 statements of analysis about the upcoming game citing specific statistics.
Include an outro with one of the following:
- A joke.
- A relatable personal anecdote (this anecdote can be fabricated).
- A discussion about the cities.
Constraints:
Do not invent any details beyond the provided information.
Do not mention any players.
Writers and Attribution:
Game summaries must be written by one of the Strong Sports Analysis Ltd's 4-star writers:
- Ace Greybeard: A former player with strong opinions and penchant for strong black and white statements.
- Sai Mitchell: The man who popularized advanced analytics in the modern NBA, and loves using the numbers to predict outcomes..
- Lisa Wolfsburgh: The winningest coach in women's basketball history, a strategic mind wrapped in a winner. Her takes are insightful and interesting.
- Alex Hoffman: The former NBA commissioner credited with laying the foundation of the modern NBA. His ideas are sharp but old fashioned.
Note:
Each game summary should carry the writer's byline.
A writer can write more than one summary, but only these four writers may author the game summaries.
Before writing the closing paragraph, ensure that all game summaries have been completed.
Editor Contributions:
The Introductory and Closing paragraphs are written by Mistro, the Editor.
The closing paragraph must reference the paragraphs written by the other writers.
Blog Post Structure:
Title:
“Ranking tonight's Games.”
Sections:
The games are organized into three sections based on probability differences:
Section 1: "Must Watch" - For games where the probabilities are less than 10% apart.
Section 2: "Close but Clear Favorite" - For games where the probabilities are between 10% and 30% apart.
Section 3: "David vs. Goliath" - For games where the probabilities are greater than 30% apart.
Game Information Format:
For each game, use the following structure:

Home Team Name vs. Away Team Name  
Home Team: [W-L: wins-losses, Win%: win_pct, SRS: srs, PS/G: pts_per_g, PA/G: opp_pts_per_g], SSA Probability: Home Probability  
Away Team: [W-L: wins-losses, Win%: win_pct, SRS: srs, PS/G: pts_per_g, PA/G: opp_pts_per_g], SSA Probability: Away Probability  
Delta: Delta between probabilities

Please write the blog post below:
"""
    # Append each game information to the prompt.
    for i, game in enumerate(ranked_games, start=1):
        home_info = game.get("home_team_info", {})
        away_info = game.get("away_team_info", {})
        prompt += f"\nGame {i}:\n"
        prompt += (
            f"Home Team: {game['home_team']} "
            f"(W-L: {home_info.get('wins', 'N/A')}-{home_info.get('losses', 'N/A')}, Win%: {home_info.get('win_pct', 'N/A')}, "
            f"SRS: {home_info.get('srs', 'N/A')}, PS/G: {home_info.get('pts_per_g', 'N/A')}, PA/G: {home_info.get('opp_pts_per_g', 'N/A')}) "
            f"with probability {game['home_prob']*100:.0f}%\n"
        )
        prompt += (
            f"Away Team: {game['away_team']} "
            f"(W-L: {away_info.get('wins', 'N/A')}-{away_info.get('losses', 'N/A')}, Win%: {away_info.get('win_pct', 'N/A')}, "
            f"SRS: {away_info.get('srs', 'N/A')}, PS/G: {away_info.get('pts_per_g', 'N/A')}, PA/G: {away_info.get('opp_pts_per_g', 'N/A')}) "
            f"with probability {game['away_prob']*100:.0f}%\n"
        )
        prompt += f"Delta: {game['delta']*100:.0f}%\n"
    prompt += "\nPlease write the blog post below:\n"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Change to your preferred model, e.g., "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are ChatGPT, a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=16384
        )
        blog_post = response["choices"][0]["message"]["content"].strip()
        return blog_post
    except Exception as e:
        print(f"Error generating blog post: {e}")
        return "Error generating blog post."

# -------------------------------
# UPDATED publish_blog_post Function Using Jinja2 Template
# -------------------------------

def publish_blog_post(blog_post, ranked_games):
    """
    Generates an HTML file that contains the game rankings and the blog post.
    This version uses a Jinja2 template (stored in the templates folder) that links to an external CSS file.
    Returns the filename of the published blog post.
    """
    # Set up the Jinja2 environment to load templates from the "templates" folder.
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('template.html')
    
    # Render the template with dynamic values.
    rendered_html = template.render(
         title="The Three Best Games to Watch Tonight",
         header="The Three Best Games to Watch Tonight",
         games=ranked_games,
         blog_post=markdown2.markdown(blog_post),
         generated_on=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"blog_post_{timestamp}.html"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(rendered_html)
        print(f"Blog post published as {os.path.abspath(filename)}")
    except Exception as e:
        print(f"Error writing blog post to file: {e}")
    return filename

# -------------------------------
# Main Program Flow
# -------------------------------

def main():
    # --- Step 1: Fetch and Process Odds Data ---
    sport_key = "basketball_nba"  # Change if needed
    print(f"Fetching odds for {sport_key}...")
    odds_data = get_odds(sport_key)
    processed_events = []
    for event in odds_data:
        if event.get('commence_time') and is_event_today(event.get('commence_time')):
            pe = process_event(event)
            if pe:
                processed_events.append(pe)
    if not processed_events:
        print("No events found for today.")
        return
    print(f"Processed {len(processed_events)} events for today.")

    # --- Step 2: Convert Processed Events into Game Data for the Blog ---
    game_data = []
    for event in processed_events:
        try:
            home_prob = event["average_home_odds"]["implied_percentage"]
            away_prob = event["average_away_odds"]["implied_percentage"]
            delta = abs(home_prob - away_prob)
            game_data.append({
                "home_team": event["home_team"],
                "away_team": event["away_team"],
                "home_prob": home_prob,
                "away_prob": away_prob,
                "delta": delta,
                "game_date": event.get("game_date")
            })
        except Exception as e:
            print(f"Error processing event for game data: {e}")
    if not game_data:
        print("No game data available.")
        return

    # --- Step 3: Enrich Game Data with Team Standings ---
    print("Fetching team standings...")
    standings = fetch_team_standings(debug=True)
    for game in game_data:
        home = game.get("home_team")
        away = game.get("away_team")
        game["home_team_info"] = standings.get(home, {})
        game["away_team_info"] = standings.get(away, {})

    ranked_games = rank_games(game_data)

    # --- Step 4: Generate and Publish the Blog Post ---
    print("Generating blog post via OpenAI...")
    blog_post = generate_blog_post(ranked_games)
    output_file = publish_blog_post(blog_post, ranked_games)
    print(f"Blog post generated and saved as {output_file}")

if __name__ == "__main__":
    main()
