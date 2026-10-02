# Content packs

Code4Scene scenes are built from third-party Unreal Engine content packs sold or
given away on [Fab](https://www.fab.com). **This repository does not include
any pack content.** To build the public dataset you get each pack from its Fab
listing yourself, install it into an Unreal Engine 5.8 project, and then run
the dataset builder (see [BUILD_DATASET.md](BUILD_DATASET.md)).

The public set (225 cases) needs **48 third-party packs** and Epic's
**Starter Content**, which ships with the engine. The machine-readable list is
[`benchmark/packs.yaml`](../benchmark/packs.yaml). It records, for each pack, the
listing, the public cases that use it, the seller demo levels that image-to-scene
ground truth was built from, and how confident we are in the listing match.

Listings were last checked on **2026-09-26**. Prices, license terms and
supported engine versions are set by the sellers and can change. The Fab listing
page is always the authoritative source.

## Licenses

Each pack is covered by the license shown on its Fab listing (Fab's Standard
License, the UE Marketplace license, or CC BY 4.0 for the Korea Heritage
Service pack). None of that content is redistributed here. Your use of a pack,
including running AI agents on scenes built from it, has to follow its license.
Each Fab listing also shows an **"Allows usage with AI"** field, copied into the
table below. Check it against your intended use before you start. CC BY 4.0
content requires attribution: credit "Korea Heritage Service".

## Installing a pack

1. **Get the listing.** Open the Fab link in the table, sign in to Fab, pick the
   license tier that fits you (Personal or Professional where offered), and
   buy it. For a free pack, click **Add to My Library**. The pack then shows up
   in your Fab library.
2. **Add it to your UE 5.8 project.** Either:
   - **Epic Games Launcher:** go to *Unreal Engine > Library > Fab Library*,
     find the pack and click **Add to Project**. Then choose your 5.8 project.
   - **Fab plugin in the editor:** open the Fab window from the UE 5.8 editor,
     go to *My Library*, and add the pack to the open project.
3. **Check the folder.** The pack has to end up at
   `<YourProject>/Content/<Folder>`, where `<Folder>` is the name in the first
   column of the table (for example `Content/Street_NY`). Task files and
   ground-truth scenes refer to assets as `/Game/<Folder>/...`, so do not rename
   or move the folder. Names are case-sensitive on Linux.

### Packs that do not list UE 5.8

26 listings do not (yet) name UE 5.8 as a supported version. For these, either:

- tick **Show all projects** in the launcher's *Add to Project* dialog, pick
  your 5.8 project, and select the build listed under "Build differences"
  below if the pack is there, otherwise the newest version the listing offers; or
- add the pack to a project on that engine version, then use
  *Asset Actions > Migrate* to copy the `<Folder>` directory into your 5.8
  project's `Content/` directory.

This applies to: `Abandoned_Psychiatric_Hospital`, `AfricanLocation`, `ArchVizInteriorVol2`, `Big_Town`, `bugchonmunhwa`, `CastleRiver`, `Chefchaouen_Village`, `CyberPunk_City_VR_Mobile`, `CyberpunkWC`, `Diner`, `Dungeon`, `EgyptianTemple`, `EuropeanStreet`, `Gas_Station`, `HorrorEnviroment`, `MedievalBuildings01`, `MedievalForgePack`, `Modular_house`, `NordicHarbour`, `OldBuilding`, `RuralAustralia`, `UrbanDecay`, `VictorianRoom`.

### Complete-project listings

5 listings are delivered as a complete project rather than as an asset
pack: `Chefchaouen_Village`, `EuropeanStreet`, `Ruins`, `ArchVizInteriorVol2`, `MiddleEasternTown`. In the launcher, use **Create Project**
(on the newest engine version offered). Then copy or migrate that project's
`Content/<Folder>` directory into your 5.8 project's `Content/` directory.

### Starter Content

`StarterContent` has no download. In the UE 5.8 editor, go to *Add > Add
Feature or Content Pack > Starter Content*, or tick **Starter Content** when you
create the project. It must be at `Content/StarterContent`.

The `OldBuilding` materials also reference
`/Game/Cabin_Pack/Master_Mat/T_MacroVariation`, which that pack does not ship.
The dataset builder creates it (as `Content/Cabin_Pack`) from Starter
Content's `T_MacroVariation` before it builds the `old-building` levels, so
there is nothing to download for `Cabin_Pack`.

## Which settings need which packs

You only need the packs for the settings you plan to build.

| Setting | Public cases | Packs | List price of those packs at check |
|---|---|---|---|
| Text-to-scene | 150 | 31 | $736.85 + 15 packs without a recorded price |
| Image-to-scene, indoor | 25 | 9 + Starter Content | $224.93 + 1 pack without a recorded price |
| Image-to-scene, outdoor | 50 | 10 + Starter Content | $385.92 + `TrainStation` (not yet identified) |
| All public cases | 225 | 48 + Starter Content | $1,310.72 + `TrainStation`, 15 text-to-scene packs and 1 image-to-scene pack without a recorded price |

Prices are Personal-tier list prices in USD, without discounts or taxes. Street New York and Russian Winter Town are used by both text-to-scene and outdoor cases, so they are counted once in the total.

- **Text-to-scene:** `Abandoned_Psychiatric_Hospital`, `AbandonedBuilding`, `AncientRuins`, `BattlegroundKit`, `Big_Town`, `bugchonmunhwa`, `CastleRiver`, `Chefchaouen_Village`, `CityCore_Paris`, `ContainerYard`, `CyberPunk_City_VR_Mobile`, `CyberPunk_street`, `EgyptianTemple`, `Elven_Valley`, `EuropeanStreet`, `FactoryEnvironment`, `Hangar`, `Hong_Kong_Street`, `MedievalBuildings01`, `MedievalScandinavianVillage`, `Modular_Gardens`, `ModularGothicFantasyEnvironment`, `NordicHarbour`, `Offworld`, `PostApocalypticMarket`, `Ruins`, `Scrapopolis`, `Street_NY`, `UrbanDecay`, `Wild_West_Race`, `WinterTown`.
- **Image-to-scene, indoor:** `ArchVizInteriorVol2`, `CyberpunkWC`, `Dungeon`, `HighSchoolClassroom`, `HorrorEnviroment`, `Laboratory_in_Loft_Style`, `MedievalForgePack`, `Office`, `StarterContent`, `VictorianRoom`.
- **Image-to-scene, outdoor:** `AfricanLocation`, `Diner`, `Gas_Station`, `MiddleEasternTown`, `Modular_house`, `OldBuilding`, `RuralAustralia`, `StarterContent`, `Street_NY`, `TrainStation`, `WinterTown`.

The per-case list is in `used_by` in [`benchmark/packs.yaml`](../benchmark/packs.yaml).

## Pack table

The last column lists settings with case counts. T2S = text-to-scene, I2S-in = image-to-scene indoor, I2S-out = image-to-scene outdoor.

| Folder (`/Game/...`) | Fab listing | Seller | License | Price at check | Allows AI use | Listed UE versions | Download type | Used by (cases) |
|---|---|---|---|---|---|---|---|---|
| `Abandoned_Psychiatric_Hospital` | [Abandoned Psychiatric Hospital](https://www.fab.com/listings/414199b4-773b-4fd1-9c58-6676d8af06d5) | ART EQUILIBRIUM | Standard License | $44.99 | Yes | 4.25-5.6 | asset pack | T2S (5) |
| `AbandonedBuilding` | [Abandoned Building](https://www.fab.com/listings/8dfc1bcd-c652-4fc8-8b2f-88df9155725b) | DevTon Studio | — | — | Yes | — | — | T2S (4) |
| `AfricanLocation` | [African desert 4x4 km2 open world location](https://www.fab.com/listings/75c21d61-c868-4cd5-aec9-e3e310cdcbce) | 3dlocatons | Standard License | $10.99 | Yes | 4.27, 5.0-5.4 | asset pack | I2S-out (4) |
| `AncientRuins` | [Modular Ancient Ruins](https://www.fab.com/listings/be34078b-d989-4a06-b956-17072740dc4f) | archafizov | — | — | Yes | — | — | T2S (4) |
| `ArchVizInteriorVol2` | [Archviz Interior vol.2](https://www.fab.com/listings/cb492691-2363-4ff9-a16b-87d6b3fb081e) | Next Level 3D | Standard License | $24.99 | Yes | 5.2-5.7 | complete project | I2S-in (6) |
| `BattlegroundKit` | [Military Battlefield Kit and Middle East Set](https://www.fab.com/listings/ea809594-f461-49c9-9dd1-32db3eb9def3) | Denys Rutkovskyi | — | — | Yes | — | — | T2S (4) |
| `Big_Town` | [Medieval Big Farm Town](https://www.fab.com/listings/b93a9f61-9220-4342-83d0-361aee394e77) | SliderOK | Standard License | $55.99 | Yes | 4.27, 5.1-5.4 | asset pack | T2S (5) |
| `bugchonmunhwa` | [KHS - House of Gyedong](https://www.fab.com/listings/ec435a53-b1ce-4f92-affb-8289e729d53e) | KOREA HERITAGE SERVICE | CC BY 4.0 | Free | Yes | 5.2-5.4 | asset pack | T2S (5) |
| `CastleRiver` | [Fantasy Medieval Castle Kit](https://www.fab.com/listings/0748515c-18e0-48bc-bcaf-d5e39e66bc46) | Denys Rutkovskyi | Standard License | $59.99 | Yes | 4.21-5.6 | asset pack | T2S (6) |
| `Chefchaouen_Village` | [Stylized Blue City – Modular Environment Kit (UE5)](https://www.fab.com/listings/5219d07d-8fea-4f62-897f-c78c680b2cc7) | Aksil Games | Standard License | $49.99 | Yes | 5.5-5.7 | complete project | T2S (5) |
| `CityCore_Paris` | [City Core - Paris](https://www.fab.com/listings/9155a337-2c74-4c5b-bff1-43adae672618) | PolySphere Studio | Standard License | $219.99 | Yes | 5.2-5.8 | asset pack | T2S (5) |
| `ContainerYard` | [Container Yard Environment Set](https://www.fab.com/listings/3a882c55-de05-4790-8d03-b1adc99da4bc) | Denys Rutkovskyi | — | — | Yes | — | — | T2S (9) |
| `CyberPunk_City_VR_Mobile` | [CyberPunk City Mobile and VR](https://www.fab.com/listings/73e1dd8c-a755-415e-89f6-b011a4bd7503) | Egor Ilin | UE Marketplace | $8.99 | Yes | 4.20-5.4 | asset pack | T2S (6) |
| `CyberPunk_street` | [Cyberpunk street assets](https://www.fab.com/listings/1f41a126-aff1-4251-bd45-e4197b6a098e) | AlexeyKuleshov | — | — | Yes | — | — | T2S (4) |
| `CyberpunkWC` | [Cyberpunk Toilet Environment Kit](https://www.fab.com/listings/19c5a0db-b413-4a27-ab0d-6036a7b842fe) | Denys Rutkovskyi | Standard License | $39.99 | Yes | 4.21-5.6 | asset pack | I2S-in (2) |
| `Diner` | [Rosie's Restaurant/Diner](https://www.fab.com/listings/dccbefcf-1f53-45c3-aae1-105bb836683f) | SoloMode | UE Marketplace | $29.99 | Yes | 4.23-5.3 | asset pack | I2S-out (6) |
| `Dungeon` | [Dungeon Environment / 135+ Assets](https://www.fab.com/listings/bb39bae4-7f7a-4127-b07e-151cf52db0f6) | PackDev | Standard License | Free | Yes | 4.26-5.4 | asset pack | I2S-in (2) |
| `EgyptianTemple` | [Egyptian Temple](https://www.fab.com/listings/e7c415c9-9e0d-4d2c-ba3f-f1c5ae537e91) | Infuse Studio | UE Marketplace | $29.99 | Yes | 4.18-4.24 | asset pack | T2S (6) |
| `Elven_Valley` | [Stylized Elven Valley](https://www.fab.com/listings/46205dbb-cfb9-45ba-814f-d12ea606b4bb) | StylArts | — | — | Yes | — | — | T2S (6) |
| `EuropeanStreet` | [European Street 3d Scene](https://www.fab.com/listings/89fe67c8-718c-4b91-84b7-3e863c3dc4e5) | 3D Happy | Standard License | $24.99 | Yes | 5.6-5.7 | complete project | T2S (5) |
| `FactoryEnvironment` | [Factory Environment Collection](https://www.fab.com/listings/2ee66462-8c2b-4303-892c-83f7fc0d9b3e) | Denys Rutkovskyi | — | — | Yes | — | — | T2S (14) |
| `Gas_Station` | [Route 66 Gas Station / Modular Environment](https://www.fab.com/listings/421cef13-0000-4b2c-9814-974bfea2790f) | PackDev | Standard License | $29.99 | Yes | 4.22-5.7 | asset pack | I2S-out (6) |
| `Hangar` | [Hangar](https://www.fab.com/listings/26114680-5785-4ba4-88d6-bd0961a485aa) | MGX Studio | — | — | Yes | — | — | T2S (5) |
| `HighSchoolClassroom` | [HighSchool Classroom](https://www.fab.com/listings/10012851-05ef-46b9-b4a2-45d8f97695e2) | ZhenyaChistovsky | — | — | Yes | — | — | I2S-in (3) |
| `Hong_Kong_Street` | [Hong Kong Street](https://www.fab.com/listings/8cce67d3-acf7-4e1d-bf14-2c2186bf439f) | ART EQUILIBRIUM | Standard License | $34.99 | Yes | 4.25-5.8 | asset pack | T2S (6) |
| `HorrorEnviroment` | [Basement environments](https://www.fab.com/listings/38709327-620f-4580-9d4f-e0b0e4ed556d) | 3Devils | Standard License | $24.99 | Yes | 4.20-5.4 | asset pack | I2S-in (1) |
| `Laboratory_in_Loft_Style` | [Laboratory Interior Studio in Loft Style](https://www.fab.com/listings/160d9e3b-982e-42bc-b55d-fb9ff3f4b46b) | Dragon Motion | Standard License | $19.99 | Yes | 4.11-4.27, 5.1-5.8 | asset pack | I2S-in (3) |
| `MedievalBuildings01` | [Medieval Buildings Volume 1](https://www.fab.com/listings/18e7c2f6-166d-4808-8ed7-7f95018e7961) | Moore Game Dev | Standard License | $19.99 | Yes | 4.21-5.4 | asset pack | T2S (5) |
| `MedievalForgePack` | [Medieval Forge Pack - VR READY](https://www.fab.com/listings/9833039f-55a2-4624-be88-9291a2864097) | Wester Games | Standard License | $9.99 | Yes | 4.22-5.6 | asset pack | I2S-in (4) |
| `MedievalScandinavianVillage` | [Medieval Scandinavian Village - Modular Environment](https://www.fab.com/listings/7eb8c2c4-373e-4ee7-bb06-1b3aff8752e1) | Lumits | — | — | Yes | — | — | T2S (5) |
| `MiddleEasternTown` | [Middle Eastern Town Gigapack w/ Level PCG ( MET, Palace, Bazaar, City)](https://www.fab.com/listings/405ffca7-aed7-4db8-ab9d-0c80e00a2d7b) | Leartes Studios | Standard License | $169.99 | Yes | 5.4-5.8 | complete project | I2S-out (7) |
| `Modular_Gardens` | [Classical Modular Gardens](https://www.fab.com/listings/deb3141f-0f08-444f-b0a3-2c177f627b92) | ScottFlinders | — | — | Yes | — | — | T2S (5) |
| `Modular_house` | [Suburban Modular House Pack](https://www.fab.com/listings/8148d0e3-3265-4b14-b2ce-6dae86f2bb19) | Alexandr Bratus | Standard License | $99.99 | Yes | 5.4, 5.5, 5.6 | asset pack | I2S-out (5) |
| `ModularGothicFantasyEnvironment` | [Modular Gothic/Fantasy Environment](https://www.fab.com/listings/92987f7a-6ff7-4884-962b-4c41dddd4cc1) | Stormcrow Studios | — | — | Yes | — | — | T2S (4) |
| `NordicHarbour` | [Nordic Harbour - Modular City Building Kit](https://www.fab.com/listings/5b9be947-566c-4a65-9fe7-ec2eeb7701bd) | Wester Games | Standard License | $49.99 | Yes | 4.22-5.6 | asset pack | T2S (5) |
| `Office` | [DownTown Office](https://www.fab.com/listings/0c920ed7-64f9-4010-a099-5374b4f79bfb) | Replex | Standard License | $79.99 | Yes | 5.0-5.8 | asset pack | I2S-in (3) |
| `Offworld` | [SciFi Environment — 'Offworld'](https://www.fab.com/listings/eb927166-57e6-4f19-9137-f072e819c93a) | Ryan Honey | — | — | Yes | — | — | T2S (5) |
| `OldBuilding` | [Old Building](https://www.fab.com/listings/bd8353ec-6997-43eb-bc3d-aa5a8902173c) | Paradox Studio | UE Marketplace | $7.99 | Yes | 4.24-5.4 | asset pack | I2S-out (6) |
| `PostApocalypticMarket` | [Post Apocalyptic Stylized Market](https://www.fab.com/listings/d60e8b48-c2e4-462a-b0bb-9f2430a932ba) | SilverSet Studios | — | — | Yes | — | — | T2S (5) |
| `Ruins` | [Ruins](https://www.fab.com/listings/e108cf47-d633-4584-a392-0eef0caaf484) | Anil Isbilir | Standard License | $49.99 | Yes | 4.20-4.27, 5.0-5.8 | complete project | T2S (6) |
| `RuralAustralia` | [Rural Australia](https://www.fab.com/listings/1c1467ce-a2f5-4be1-8988-9069f90a8571) | Andrew Svanberg Hamilton | UE Marketplace | Free | Yes | 4.26-4.27, 5.0-5.4 | asset pack | I2S-out (3) |
| `Scrapopolis` | [Scrapopolis - Steampunk village](https://www.fab.com/listings/fa19fa3a-62e4-4528-82dc-19bc66521f3b) | ARK.KRA | — | — | Yes | — | — | T2S (4) |
| `StarterContent` | Starter Content (ships with the engine) | Epic Games | Unreal Engine EULA (engine content) | Free | n/a | 5.8 | engine feature pack | I2S-in (2), I2S-out (6) |
| `Street_NY` | [Street New York](https://www.fab.com/listings/d327e87a-7aaf-4e4e-b3a9-69e98e0bc25a) | ART EQUILIBRIUM | Standard License | $34.99 | Yes | 4.20-5.8 | asset pack | T2S (5), I2S-out (5) |
| `TrainStation` | **Not yet identified** (see below) | ? | ? | ? | Yes | ? | ? | I2S-out (5) |
| `UrbanDecay` | [Post-Industrial Environment](https://www.fab.com/listings/060b1fea-0ed5-4172-b59d-165ea89ae606) | Joshua Giles | UE Marketplace | $49.99 | Yes | 4.25-5.1 | asset pack | T2S (5) |
| `VictorianRoom` | [Victorian Dining Room](https://www.fab.com/listings/39fb2fee-f389-49bf-a90a-408d0481dc63) | Infuse Studio | UE Marketplace | $24.99 | Yes | 4.13-4.24 | asset pack | I2S-in (1) |
| `Wild_West_Race` | [Wild West Racing Town](https://www.fab.com/listings/c6d14a6f-a7d7-4d14-b757-35dfa31e05af) | Sonicbrostudio | — | — | Yes | — | — | T2S (4) |
| `WinterTown` | [Russian Winter Town](https://www.fab.com/listings/5bd7045e-b0ae-45a4-ab00-72b2060ab4c5) | GeorgeShachnev | Standard License | $1.99 | Yes | 4.19-5.8 | asset pack | T2S (6), I2S-out (3) |

## Known caveats

- **`TrainStation` has not been matched to a Fab listing yet.** This folder is
  used by 5 outdoor cases. It holds a modern urban train station (overpass,
  skyscrapers, branded signs, overhead-wire splines) with the levels
  `TrainStation`, `TrainStation_Optimised`, `Demonstration`, `Overview`,
  `Cinematography_Level` and `Sound_Level`. It is *not* "Victorian Train Station
  and Railroad Modular Set" or "Modular Train Station Environment". Until a
  listing is published here, those 5 cases cannot be built.
- **Similar names, different products.** Take the listing from the table, not
  from a Fab search. `OldBuilding` is *Old Building* by Paradox
  Studio. `Ruins` is *Ruins* by Anil Isbilir (folder `/Game/Ruins`;
  `/Game/AncientRuins` is a different pack).
- **Folder names that do not match the product name:** `CastleRiver` (*Fantasy
  Medieval Castle Kit*), `HorrorEnviroment` (*Basement environments*),
  `UrbanDecay` (*Post-Industrial Environment*), `Chefchaouen_Village`
  (*Stylized Blue City*), `bugchonmunhwa` (*KHS - House of Gyedong*; the case id
  `khs-house-of-baeryeom` is historical), `Office` (*DownTown Office*), `Diner`
  (*Rosie's Restaurant/Diner*), `Gas_Station` (*Route 66 Gas Station*),
  `Modular_house` (*Suburban Modular House Pack*).
- **Build differences.** The benchmark was made from specific builds of some
  packs, recorded in each pack's `notes` in `packs.yaml`. Install these builds
  where the listing still offers them:
  `Laboratory_in_Loft_Style` (UE4 build; the 5.1+ build was reworked for
  Lumen), `Gas_Station` (UE 4.22), `OldBuilding` (UE 4.24), `Big_Town`
  (UE 5.1), `Modular_house` (UE 5.6). `Street_NY` and
  `Abandoned_Psychiatric_Hospital` were also older builds with fewer
  textures than the current listings. With another build, an asset that a
  task names may be missing or look different, and `verify` can report a
  mismatch for the image-to-scene scenes built from that pack. If a
  `/Game/<Folder>/...` asset is reported missing, check which build of the pack
  you installed.
