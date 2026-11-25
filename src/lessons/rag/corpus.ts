// authored-by: Claude Opus 5.5
/**
 * A small, entirely fictional knowledge base: the support pages for the "Kestrel K2", a home
 * espresso machine, and the shop that sells it. No language model has ever seen these facts, which
 * is the point: a model can only answer questions about them if the right text is put in front of it.
 *
 * Each question is labelled with the one sentence that answers it, so retrieval can be scored exactly.
 */
export type Doc = { id: string; title: string; sentences: string[] };

export const DOCS: Doc[] = [
  {
    id: 'warranty',
    title: 'Warranty',
    sentences: [
      'Every Kestrel K2 comes with a two-year warranty from the date of delivery.',
      'The warranty covers faults in materials and workmanship, including the pump, boiler and control board.',
      'Damage caused by limescale is not covered, so regular descaling is a condition of the warranty.',
      'To make a claim, contact support with your order number and a short video of the fault.',
      'Repairs under warranty are free, including courier collection and return.',
    ],
  },
  {
    id: 'returns',
    title: 'Returns and refunds',
    sentences: [
      'You can return an unused machine within 30 days of delivery for a full refund.',
      'Machines that have been used can be returned within 14 days, minus a cleaning fee of 25 euros.',
      'Refunds are paid to the original card within five working days of the machine arriving at our warehouse.',
      'Accessories bought separately, such as milk jugs and tampers, can be returned within 30 days if unopened.',
    ],
  },
  {
    id: 'shipping',
    title: 'Delivery',
    sentences: [
      'Orders placed before 2 pm on a weekday leave our warehouse the same day.',
      'Standard delivery takes two to four working days and is free on orders over 100 euros.',
      'Express delivery arrives the next working day and costs 9 euros.',
      'We currently deliver to the EU, Norway, Switzerland and the United Kingdom.',
      'Every parcel has a tracking link, sent by email when it leaves the warehouse.',
    ],
  },
  {
    id: 'descale',
    title: 'Descaling',
    sentences: [
      'Descale the K2 every two months, or monthly if your tap water is hard.',
      'Use only Kestrel descaling liquid; vinegar can damage the seals.',
      'Mix one sachet with one litre of water in the tank and hold the steam and hot water buttons together for five seconds.',
      'The cycle takes about 25 minutes and the descale light flashes until it is finished.',
      'Afterwards, rinse the tank and run two full tanks of clean water through the machine.',
    ],
  },
  {
    id: 'cleaning',
    title: 'Daily cleaning',
    sentences: [
      'Wipe the steam wand with a damp cloth immediately after every use, before milk dries on it.',
      'Purge the steam wand for two seconds to clear milk from the tip.',
      'Empty the drip tray when the red float appears.',
      'The drip tray, water tank and portafilter basket are dishwasher safe; the steam wand is not.',
      'Once a week, backflush the group head with the blind basket and a cleaning tablet.',
    ],
  },
  {
    id: 'grind',
    title: 'Grind size and dose',
    sentences: [
      'For a double shot, use 18 grams of coffee ground fine, about like table salt.',
      'If the shot runs faster than 20 seconds and tastes sour, grind finer.',
      'If the shot runs slower than 35 seconds and tastes bitter, grind coarser.',
      'Pre-ground supermarket coffee is usually too coarse for espresso and gives weak, watery shots.',
    ],
  },
  {
    id: 'milk',
    title: 'Steaming milk',
    sentences: [
      'Fill the jug to just below the spout with cold milk straight from the fridge.',
      'Keep the wand tip just under the surface at first to add air, then lower it to spin the milk.',
      'Stop when the jug is too hot to hold comfortably, around 60 degrees.',
      'Oat and other plant milks foam best when they are labelled barista edition.',
    ],
  },
  {
    id: 'errors',
    title: 'Error codes',
    sentences: [
      'E1 means the water tank is empty or not seated properly.',
      'E2 means the boiler is overheating; switch the machine off and let it cool for 30 minutes.',
      'E3 means the pump cannot draw water, usually because of an air bubble; run the hot water button for ten seconds.',
      'E7 means the descaling cycle was interrupted; start it again from the beginning.',
      'E9 means a sensor fault that cannot be fixed at home; contact support.',
    ],
  },
  {
    id: 'nocoffee',
    title: 'No coffee comes out',
    sentences: [
      'If the pump runs but no coffee comes out, the coffee may be ground too fine or tamped too hard.',
      'Try a coarser grind and a lighter tamp first.',
      'If water comes out of the group head without a portafilter, the problem is the coffee, not the machine.',
      'If no water comes out at all, the machine may need descaling.',
    ],
  },
  {
    id: 'leak',
    title: 'Leaks',
    sentences: [
      'Water under the machine usually means the drip tray is full or not pushed fully in.',
      'Water around the portafilter during a shot means the group gasket is worn and should be replaced every 12 months.',
      'A gasket replacement kit costs 12 euros and includes a tool and instructions.',
      'A leak from the bottom of the machine while it is switched off needs a repair; contact support.',
    ],
  },
  {
    id: 'power',
    title: 'Energy and standby',
    sentences: [
      'The K2 heats up in about 45 seconds thanks to its thermoblock.',
      'It switches itself to standby after 20 minutes without use, which you can change to 10 or 60 minutes in the settings.',
      'In standby it uses less than half a watt.',
      'The machine is rated at 1,450 watts and should be plugged directly into a wall socket, not an extension lead.',
    ],
  },
  {
    id: 'specs',
    title: 'Specifications',
    sentences: [
      'The K2 is 24 centimetres wide, 33 centimetres deep and 31 centimetres tall.',
      'It weighs 6.2 kilograms empty.',
      'The removable water tank holds 1.8 litres.',
      'The pump is rated at 15 bar, and the machine brews at 9 bar.',
      'It is available in matte black, cream and steel.',
    ],
  },
  {
    id: 'app',
    title: 'The Kestrel app',
    sentences: [
      'The Kestrel app connects to the K2 over Bluetooth and lets you set the shot temperature and volume.',
      'It reminds you when descaling is due, based on how many shots you have made.',
      'Firmware updates are installed through the app and take about three minutes.',
      'The app is free on iOS and Android and is not needed to use the machine.',
    ],
  },
  {
    id: 'parts',
    title: 'Spare parts',
    sentences: [
      'Spare portafilters, baskets, gaskets and water tanks can be ordered from the parts page.',
      'The single-wall baskets are for freshly ground coffee; the pressurised baskets forgive pre-ground coffee.',
      'Replacement water filters cost 8 euros each and should be changed every two months.',
    ],
  },
  {
    id: 'account',
    title: 'Your account',
    sentences: [
      'You can change your delivery address until the order leaves the warehouse.',
      'Invoices are available as PDF downloads under Orders in your account.',
      'To delete your account, write to privacy@kestrel.example; we confirm within seven days.',
    ],
  },
  {
    id: 'safety',
    title: 'Safety',
    sentences: [
      'Never open the casing; there are no parts inside that you can service yourself.',
      'The steam wand and group head get hot enough to burn; keep children away while the machine is on.',
      'Always switch the machine off and unplug it before cleaning anything other than the steam wand.',
    ],
  },
];

/** A question, the sentence that answers it, and what kind of wording it uses. */
export type Question = { q: string; doc: string; sent: number; kind: 'paraphrase' | 'exact' | 'plain' };

export const QUESTIONS: Question[] = [
  { q: 'How long is the guarantee on the K2?', doc: 'warranty', sent: 0, kind: 'paraphrase' },
  { q: 'Will you fix it for free if the pump breaks?', doc: 'warranty', sent: 1, kind: 'paraphrase' },
  { q: 'Is chalk build-up covered by the warranty?', doc: 'warranty', sent: 2, kind: 'paraphrase' },
  { q: 'Can I send back a machine I have already used?', doc: 'returns', sent: 1, kind: 'paraphrase' },
  { q: 'How soon do I get my money back?', doc: 'returns', sent: 2, kind: 'paraphrase' },
  { q: 'What does next-day shipping cost?', doc: 'shipping', sent: 2, kind: 'paraphrase' },
  { q: 'Do you ship to Switzerland?', doc: 'shipping', sent: 3, kind: 'plain' },
  { q: 'How often should I descale?', doc: 'descale', sent: 0, kind: 'plain' },
  { q: 'Can I use vinegar to remove limescale?', doc: 'descale', sent: 1, kind: 'plain' },
  { q: 'How do I start the descaling cycle?', doc: 'descale', sent: 2, kind: 'plain' },
  { q: 'Can I put the steam wand in the dishwasher?', doc: 'cleaning', sent: 3, kind: 'plain' },
  { q: 'How much ground coffee goes into a double?', doc: 'grind', sent: 0, kind: 'paraphrase' },
  { q: 'My espresso tastes sour and pours too quickly. What should I change?', doc: 'grind', sent: 1, kind: 'paraphrase' },
  { q: 'My shot tastes bitter and takes forever. What should I change?', doc: 'grind', sent: 2, kind: 'paraphrase' },
  { q: 'Why is my coffee weak when I use supermarket coffee?', doc: 'grind', sent: 3, kind: 'plain' },
  { q: 'How hot should the milk get?', doc: 'milk', sent: 2, kind: 'paraphrase' },
  { q: 'Which oat milk froths well?', doc: 'milk', sent: 3, kind: 'paraphrase' },
  { q: 'What does E2 mean?', doc: 'errors', sent: 1, kind: 'exact' },
  { q: 'The display shows E3.', doc: 'errors', sent: 2, kind: 'exact' },
  { q: 'What is error E7?', doc: 'errors', sent: 3, kind: 'exact' },
  { q: 'What should I do about E9?', doc: 'errors', sent: 4, kind: 'exact' },
  { q: 'The pump runs but nothing comes out.', doc: 'nocoffee', sent: 0, kind: 'plain' },
  { q: 'Water drips around the portafilter while brewing.', doc: 'leak', sent: 1, kind: 'plain' },
  { q: 'How much is a new gasket?', doc: 'leak', sent: 2, kind: 'paraphrase' },
  { q: 'How long does the machine take to warm up?', doc: 'power', sent: 0, kind: 'paraphrase' },
  { q: 'How much electricity does it use when idle?', doc: 'power', sent: 2, kind: 'paraphrase' },
  { q: 'Can I use an extension cord?', doc: 'power', sent: 3, kind: 'paraphrase' },
  { q: 'How heavy is the K2?', doc: 'specs', sent: 1, kind: 'paraphrase' },
  { q: 'How big is the water reservoir?', doc: 'specs', sent: 2, kind: 'paraphrase' },
  { q: 'How do I update the firmware?', doc: 'app', sent: 2, kind: 'plain' },
  { q: 'Do I need the app to make coffee?', doc: 'app', sent: 3, kind: 'plain' },
  { q: 'Which basket should I use with pre-ground coffee?', doc: 'parts', sent: 1, kind: 'plain' },
  { q: 'Where can I download my invoice?', doc: 'account', sent: 1, kind: 'plain' },
  { q: 'Can I take the cover off to repair it myself?', doc: 'safety', sent: 0, kind: 'paraphrase' },
];
