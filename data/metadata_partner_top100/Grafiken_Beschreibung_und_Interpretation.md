# Grafiken: Beschreibung und Interpretation

Dieses Dokument begleitet die Auswertung **Metadaten-Kreuztabellen und P_k (partner-gewichtet, TOP-100)**. Grundlage ist jeweils **ein Partner = eine Beobachtung**: Die zehn Funktionsgewichte werden pro Korrespondent über alle Briefe in `brief_predictions_top100.csv` gemittelt; Kreuztabellen zählen daher **Partner**, nicht Einzelbriefe. Leere Codebook-Felder erscheinen als `(fehlend)`, Partner ohne exakten `Vollname`-Treffer im Codebook als `(nicht im Codebook)` — meist Schreibvarianten oder Sonderfälle. Wer die Grafiken **ohne** diese Zeile/Spalte will, führt das Skript mit `--nur-mit-codebook` aus; betroffene Namen stehen in `partners_nicht_im_codebook.csv`.

**Gestapelte Balken** (`plot_crosstab_*_row_norm_stacked`, `plot_crosstab_*_col_norm_stacked`): Zeilen- bzw. Spaltenanteile als **horizontale 100-%-Balken** (lesbarer als die blaue Heatmap bei vielen Kategorien). Die Zählmatrix wird um Zeilen/Spalten **„(nicht im Codebook)“** reduziert, die Anteile **neu berechnet** (ohne diesen Hinweis in der Diagrammüberschrift). Legenden nutzen für die Dimension **Beruf** die Bezeichnung **„Berufe/Rollen“**. Die **P_k**-Heatmaps im gleichen Ordner nutzen ebenfalls nur Partner mit Codebook-Zeile. Die Farbintensität in den Heatmaps ist **deskriptiv**; kleine Zellenbesetzungen erlauben keine belastbare Inferenz, wohl aber **Plausibilisierungen** und **Fallauswahl**.

---

## Kreuztabellen — `plot_crosstab_Nat_Bez_*`

### `plot_crosstab_Nat_Bez_counts.png` / `.pdf`

Die Grafik zeigt eine **Heatmap der absoluten Häufigkeiten**: Wie viele Korrespondenten gleichzeitig einer **Nationalität** (Zeilen) und einem **Beziehungstyp** (Spalten) zugeordnet sind. Dunklere Blautöne bedeuten höhere Besetzungszahlen. Sie macht sichtbar, **wo das TOP-100-Netz strukturell „sitzt“** — etwa ob deutschsprachige Kontakte überwiegend beruflich oder familiär kodiert sind. Leere oder sehr helle Felder weisen auf fehlende Kombinationen oder geringes **n** hin. Die Darstellung ist ideal, um **Verteilungsformen** zu erfassen, nicht um Anteile innerhalb einer Zeile oder Spalte zu vergleichen; dafür dienen die normierten Varianten. Zusammen mit der CSV `crosstab_Nat_Bez_counts.csv` lässt sich jede Zelle exakt nachlesen.

### `plot_crosstab_Nat_Bez_row_norm.png` / `.pdf`

Hier sind die **Zeilen auf 1 normiert**: Jede Zeile (Nationalität) zeigt, **welcher Anteil der Partner dieser Nationalität** auf welchen Beziehungstyp entfällt. Die Skala reicht von 0 bis 1; in der Grafik ist die Farbskala entsprechend begrenzt. So wird sichtbar, ob etwa eine Gruppe **überwiegend beruflich** oder **stärker familiär/freundschaftlich** erfasst ist — unabhängig davon, wie groß die Gruppe insgesamt ist. Das ist die passende Lesart, wenn man **„innerhalb einer Nationalität“** frägt, nicht „welche Nationalität dominiert im Netz“. Vorsicht bei Nationalitäten mit nur einem Partner: dort steht in der Zeile eine **1,0** in genau einem Feld und sonst Nullen — das ist **trivial**, nicht strukturell.

### `plot_crosstab_Nat_Bez_col_norm.png` / `.pdf`

Diese Heatmap **normiert spaltenweise auf 1**: Sie beantwortet, **welcher Anteil aller Partner eines Beziehungstyps** welcher Nationalität zugeordnet ist. Sie eignet sich, um zu sehen, ob z. B. der Typ **„Geschäftlich“** stärker deutsch- oder französisch-lastig ist — also eine **Mischungsfrage im jeweiligen Rollenfeld**. Unterschiede zu `row_norm` sind erwartbar, weil Zeilen- und Spaltenperspektiven verschiedene Fragen kodieren. Auch hier sind kleine **n** in einzelnen Spalten vorsichtig zu lesen. Für Text und Argumentation lohnt es, **beide Normierungen** nebeneinander zu verwenden, statt nur eine.

---

## Kreuztabellen — `plot_crosstab_Nat_Beruf_*`

### `plot_crosstab_Nat_Beruf_counts.png` / `.pdf`

Die Matrix bildet **Nationalität × Beruf/Rolle** ab; die Zellen enthalten Partnerzahlen. Weil **Beruf/Rolle** viele fein unterteilte Kategorien hat, wird die Grafik **groß und textdicht**; sie ist dennoch wertvoll als **Landkarte der beruflichen Verankerung** nach Zugehörigkeitsraum. Hohe Werte zeigen Kombinationen, in denen sich mehrere Korrespondenten wiederfinden (z. B. deutsch + Verleger). Sehr kleine oder leere Felder sollten nicht überinterpretiert werden. Die Grafik unterstützt die **Auswahl exemplarischer Fälle** (welche Berufsrolle in welcher Nationalszene häufig vorkommt).

### `plot_crosstab_Nat_Beruf_row_norm.png` / `.pdf`

**Zeilenanteile** zeigen pro Nationalität, **wie die beruflichen Rollen** innerhalb dieser Gruppe verteilt sind — ob also etwa französisch kodierte Partner **überwiegend Schriftsteller, Publizisten oder Verleger** sind. Das entspricht der Frage nach der **inneren Rollenstruktur** einer Nationalschicht im Sample. Bei seltenen Nationalitäten entstehen optisch **schmale, aber konzentrierte** Muster. Die Darstellung hilft, essentialisierende Sprache zu vermeiden: Es geht um **Kodierungen im Codebook**, nicht um „typische Berufe von Nation X“ im statistischen Sinn.

### `plot_crosstab_Nat_Beruf_col_norm.png` / `.pdf`

**Spaltenanteile** beantworten: **Welche Nationalitäten** tragen zu einer gegebenen **Berufsrolle** bei — etwa wie stark **„Verleger“** deutsch vs. französisch besetzt ist. Das ist die Perspektive **„Berufsfeld als Fenster zur Internationalität“**. Unterschiede zwischen Spalten spiegeln **unterschiedliche institutionelle Anbindungen** wider (Verlagswesen, Publizistik, Salon etc.). Wo **n** pro Spalte klein ist, sind Konzentrationen **descriptiv**, nicht repräsentativ.

---

## Kreuztabellen — `plot_crosstab_Nat_JH_*`

### `plot_crosstab_Nat_JH_counts.png` / `.pdf`

Hier stehen **Nationalität** und **jüdische Herkunft** (ja/nein/fehlend) in Beziehung. Die Matrix ist **kleiner** als bei Beruf; sie zeigt, wie Partner im TOP-100-Stratum **gleichzeitig** entlang beider Merkmale fallen. Sie eignet sich, **Überschneidungen** sichtbar zu machen (z. B. ob bestimmte Nationalitätslabels häufiger mit „ja“ kombiniert sind). Die Grafik ersetzt **keine** inferenzstatistische Prüfung; sie strukturiert nur die **Kodierungslage** im Korpus.

### `plot_crosstab_Nat_JH_row_norm.png` / `.pdf`

Zeilenweise Normierung zeigt, **innerhalb einer Nationalitätsgruppe** welcher Anteil mit **ja/nein** zur jüdischen Herkunft markiert ist — sofern das Feld befüllt ist. Das unterstützt eine **vorsichtige Kontextualisierung** ohne Gruppenvergleiche auf Essentialisierung zu verkürzen. Kleine Zeilen **n** führen zu extremen Anteilen; das sollte im Fließtext erwähnt werden.

### `plot_crosstab_Nat_JH_col_norm.png` / `.pdf`

Spaltenweise Lesart: **Welche Nationalitätsanteile** tragen zur Gruppe **„ja“** bzw. **„nein“** bei? Das ist relevant, wenn man die Variable **als Kontextmarker** im **international gestuften Feld** diskutiert. Die Grafik zeigt **Komposition**, nicht „Überrepräsentation“ im kausalen Sinn.

---

## Kreuztabellen — `plot_crosstab_Bez_Beruf_*`

### `plot_crosstab_Bez_Beruf_counts.png` / `.pdf`

Diese Heatmap verknüpft **Beziehungstyp** (familiär, freundschaftlich, beruflich, geschäftlich, Bekannter) mit **Beruf/Rolle**. Sie visualisiert, **welche sozialen Rollen** im Codebook **welchen Beziehungslogiken** zugeordnet sind — etwa ob **„Familie“** fast ohne spezifische Berufsbezeichnung auskommt, während **„Beruflich“** stark mit Publizist/Verleger/Schriftsteller besetzt ist. Die Grafik ist **zentral für die Diskussion von Rollenüberlagerung** vs. Trennschärfe. Sehr große Matrizen erfordern ggf. Zoom beim Lesen.

### `plot_crosstab_Bez_Beruf_row_norm.png` / `.pdf`

Pro **Beziehungstyp** wird sichtbar, **wie Berufsrollen innerhalb dieser Beziehungsform** verteilt sind — die Frage nach der **inneren Heterogenität** von z. B. „Geschäftlich“ vs. „Freund“. Das hilft, Textpassagen zu stützen, in denen **geschäftliche Kontakte** als **funktional breit** beschrieben werden (viele Rollenfacetten). Einzelne Zeilen mit **n = 1** erzeugen triviale Balken/Heatmap-Zeilen.

### `plot_crosstab_Bez_Beruf_col_norm.png` / `.pdf`

Spaltenperspektive: **Welcher Beziehungstyp** dominiert bei einer gegebenen **Berufsrolle** — etwa ob **Verleger** überwiegend **geschäftlich** oder auch **freundschaftlich** kodiert sind. Das ist eine Lesart **„Beruf als soziale Brille auf die Beziehungsform“** und eignet sich für **Fallvergleiche** (Campe, Verlegerkontakte etc.).

---

## Kreuztabellen — `plot_crosstab_Bez_JH_*`

### `plot_crosstab_Bez_JH_counts.png` / `.pdf`

Matrix **Beziehungstyp × jüdische Herkunft** mit absoluten Partnerzahlen. Sie zeigt knapp, ob bestimmte Beziehungsformen **häufiger** mit einem gesetzten Herkunftsmarker einhergehen — immer im Rahmen der **Kodierungspraxis**. Die Grafik ist übersichtlich und gut für **knappe empirische Absätze** nutzbar.

### `plot_crosstab_Bez_JH_row_norm.png` / `.pdf`

Innerhalb jedes **Beziehungstyps**: Anteil **ja/nein/(fehlend)**. Das erlaubt eine **nuancierte Einordnung**, ohne starke Behauptungen: Es geht um **Verteilung der Marker**, nicht um Identitätszuschreibungen.

### `plot_crosstab_Bez_JH_col_norm.png` / `.pdf`

Aus Sicht von **ja/nein**: Welche **Beziehungstypen** tragen wie stark bei? Das kann **illustrieren**, ob die Variable eher **gleichmäßig** über Beziehungsformen streut oder **gebündelt** vorkommt — mit allen Vorbehalten kleiner **n**.

---

## Kreuztabellen — `plot_crosstab_Beruf_JH_*`

### `plot_crosstab_Beruf_JH_counts.png` / `.pdf`

**Beruf/Rolle × jüdische Herkunft** mit Partnerzahlen. Die Grafik kann zeigen, ob bestimmte **Berufsfelder** im Sample häufiger mit einem gesetzten Marker verknüpft sind. Aufgrund vieler **Berufszeilen** ist die Heatmap **breit**; sie dient eher der **Exploration** und **Stichwortgeber** für Fallstudien als einem schnellen Urteil.

### `plot_crosstab_Beruf_JH_row_norm.png` / `.pdf`

Pro **Berufsrolle**: Verteilung von **ja/nein** — sofern die Zeile genügend Partner hat. Hier sind **Stichprobenwarnungen** besonders wichtig: Einzelberufe mit **n = 1 oder 2** erzeugen **0/1-Anteile**, die nicht verallgemeinert werden sollten.

### `plot_crosstab_Beruf_JH_col_norm.png` / `.pdf`

Aus Sicht der Spalten: **Welche beruflichen Profile** tragen zu **ja** bzw. **nein** bei? Das kann **hypothesenbildend** für weiterführende qualitative Arbeit sein, nicht für **Essentialisierung**.

---

## P_k-Profile — Nationalität

### `plot_pk_mittel_Nationalität.png` / `.pdf`

Die Heatmap zeigt das **mittlere P_k** (Mittel der zehn Kategoriegewichte über Partner) **je Nationalitätsgruppe**; die Spalten sind die **zehn Funktionskategorien**. Sie beantwortet: **Wie stark** sind INTELLECTUAL, EDITORIAL, FAMILIAL usw. **im Durchschnitt** bei Partnern dieser Nationalität — nach **partner-gewichteter** Logik. Farbintensität (Farbskala „Mittleres Gewicht“) macht **profiltypische Schwerpunkte** sichtbar, etwa ob eine Gruppe **höhere institutionelle oder editoriale Anteile** hat. Seltene Nationalitäten mit wenigen Partnern liefern **instabile Mittelwerte**; dort sind **Einzelbefunde** und **n** in der CSV zu konsultieren. Die Grafik ist **komplementär** zu den Kreuztabellen: sie spricht **Funktionsmischung**, nicht nur Häufigkeit von Metadaten.

### `plot_pk_kategoriemasse_anteil_Nationalität.png` / `.pdf`

Hier stehen **Zeilen = Funktionskategorien**, **Spalten = Nationalitäten**; jede **Zeile summiert sich zu 1**. Interpretation: Von der **gesamten „Masse“** dieser Funktion im Netz (Summe der mittleren Gewichte über alle Partner) entfällt **welcher Anteil** auf welche Nationalität? Das ist die **umgekehrte Perspektive** zu `plot_pk_mittel_*`: nicht „Profil pro Gruppe“, sondern **„wo sitzt diese Funktion im Netz geographisch-kodierungsmäßig?“**. Stark besetzte Nationalitäten (z. B. deutsch, französisch) dominieren oft mehrere Zeilen **aus Mengengründen**, nicht nur aus Profilunterschieden — das sollte textlich mitbedacht werden.

---

## P_k-Profile — Beziehungstyp

### `plot_pk_mittel_Beziehungstyp.png` / `.pdf`

Mittleres **P_k** je **Beziehungstyp** (Familie, Freund, Beruflich, Geschäftlich, Bekannter): Welche **funktionale Signatur** hat die jeweilige soziale Form im Mittel? Typisch sind **höhere FAMILIAL/TRANSACTIONAL** in bestimmten Gruppen oder **breitere** Muster bei geschäftlich/beruflich, wenn viele Dimensionen kookkurrieren. Die Grafik stützt Textpassagen zur **Rollenüberlagerung** und zur **Unterschiedlichkeit** zwischen „privaten“ und „öffentlichen“ Beziehungsformen — immer als **Mittel über Kodierungen**, nicht als Wesensurteil über Personen.

### `plot_pk_kategoriemasse_anteil_Beziehungstyp.png` / `.pdf`

Pro Funktionskategorie: **Anteil der Kategoriensumme**, der auf jeden **Beziehungstyp** entfällt (Zeilen summieren zu 1). Sie zeigt, **wo** etwa **EDITORIAL** oder **AFFECTIVE** im Netz **sozial verankert** sind — bei Familie, bei Verlagskontakten usw. Das ist hilfreich, um **Makro-Muster** mit **Dyaden-Lesarten** zu verbinden.

---

## P_k-Profile — Beruf/Rolle

### `plot_pk_mittel_Beruf.png` / `.pdf`

Viele Zeilen (einzelne Berufsrollen) × zehn Kategorien: das **mittlere funktionale Profil pro Beruf** im TOP-100-Sample. Auffällige Zeilen (z. B. Verleger, Publizist, Komponist) können **hohe Diversität oder spezifische Schwerpunkte** zeigen. **Kleines n pro Zeile** macht viele Felder **interpretationsanfällig**; die Grafik ist am ehesten ein **Raster für Fallauswahl** und **literaturgeschichtliche Einordnung** (Salon, Verlag, Politik). Extreme Farben bei **n = 1** sind **Einzelwerte**, keine Populationsparameter.

### `plot_pk_kategoriemasse_anteil_Beruf.png` / `.pdf`

Für jede **Funktionskategorie** die **Aufteilung der Gesamtmasse** auf **Berufsrollen** (Zeile summiert zu 1). Sie beantwortet: Trägt etwa **EDITORIAL** stärker über **Verleger** und **Publizisten** als über andere Rollen? Dominante Berufe im Sample können **anteilig** stark wirken, ohne dass das Profil „exklusiv“ wäre. **Vergleich** mit `plot_pk_mittel_Beruf` lohnt sich: Profil vs. Massenverteilung.

---

## P_k-Profile — Jüdische Herkunft

### `plot_pk_mittel_Juedische_Herkunft.png` / `.pdf`

Mittleres **P_k** für die Ausprägungen **ja**, **nein**, **(fehlend)** — soweit im Sample vorhanden. Unterschiede sollten **vorsichtig** gelesen werden: Die Variable ist ein **Codebook-Marker**, keine vollständige Erfassung historischer Identität. Kleine Gruppenunterschiede in **H** oder in Einzelkategorien sind **deskriptiv**; sie rechtfertigen keine kulturellen **Typologien**. Die Grafik eignet sich, **ähnliche Profilbreiten** zu zeigen oder **Hypothesen** für kontextuelle Fallarbeit zu formulieren.

### `plot_pk_kategoriemasse_anteil_Juedische_Herkunft.png` / `.pdf`

Verteilung der **Kategoriensummen** auf **ja/nein/(fehlend)**. Sie zeigt, ob bestimmte Funktionen **stärker** bei einer der Gruppen „landen“ — **anteilig** über alle Partner. Die Lesart bleibt **strukturell** (Netzwerk- und Kodierungslage), nicht **essentialisierend**. Kombination mit den **Kreuztabellen** zu Nationalität/Beziehungstyp vermeidet isolierte Deutung nur eines Merkmals.

---

## Schlussbemerkung

Alle Grafiken sind auf **dieselbe Aggregationslogik** gestützt und sollten **im Kontext** von Phasenanalysen, globalen **P_k**-Heatmaps (Phase × Richtung) und **qualitativer Lektüre** gelesen werden. Wo **n** klein ist, dienen sie **Exploration und Illustration**; wo **n** größer ist, **stützen** sie **robustere** Aussagen über **Mischung und Einbettung** von Kommunikation im Heine-Netz.
